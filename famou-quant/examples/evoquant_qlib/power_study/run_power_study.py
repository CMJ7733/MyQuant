"""Power study — 门控统计功效实测 (W1 最后一项, 第一个 Go/No-Go 门).

问题: 1 年密封晋升窗口 (~240 交易日) 内, 配对日度 RankIC 差 d_t 能可靠检出多小的真实改进 Δ?
纸面估算 std(d_t)≈0.03-0.05 → Δ_min≈0.008. 本脚本实测它, 并测 d_t 自相关对有效样本量的侵蚀.

设计 (全部在冻结协议 E1 上, E1 是 development episode, 允许自由使用):
  - 数据: Alpha158/csi300, E1 train(2008-2013)/dev(2014, early stopping)/promo(2015, 测量段),
    embargo=2 交易日尾部修剪, 预处理只在 train 段拟合 —— 与 protocol_b/splits.yaml 逐条一致.
  - 模型对三类:
      NULL    同配置不同 seed 的官方 LGBM ×5 → C(5,2)=10 对, Δ=0 零假设 (校准假阳性)
      PERTURB 官方配置单超参微扰 ×4 vs base       (模拟演化中的微小真实差异)
      GAP     浅树弱模型 vs base                  (已知大差距, 检验不误杀)
  - 每对统计: mean/std(d_t), lag-1 自相关 ρ, AR(1) 有效样本 N_eff=N(1-ρ)/(1+ρ),
    最小可检测 Δ_min = (z_.95+z_.80)·std/√N_eff (单侧 α=0.05, power=0.8),
    以及配对相对独立样本设计的方差缩减比.

Go/No-Go 判据 (评审文档 §7-W1): NULL 对中位 Δ_min ≤0.010 → GO; >0.015 → NO-GO 需改证据构造.

运行: conda run -n quant --no-capture-output python run_power_study.py
产出: rankic_promo_E1/<model>.csv (逐模型日度 RankIC), results.json, stdout 汇总表.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # 同 smoke_qlib.py: mlflow 新版禁文件后端

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_IC = HERE / "rankic_promo_E1"
OUT_IC.mkdir(exist_ok=True)

PROVIDER_URI = "/root/.qlib/qlib_data/cn_data"
MARKET = "csi300"
EMBARGO_DAYS = 2  # = label.forward_depth_days, 推导值 (protocol_b/splits.yaml)

# E1 名义边界 (splits.yaml 冻结 v1)
E1 = {
    "train": ("2008-01-01", "2013-12-31"),
    "dev": ("2014-01-01", "2014-12-31"),
    "promo": ("2015-01-01", "2015-12-31"),
}

# 官方 LightGBM/Alpha158 超参 (workflow_config_lightgbm_Alpha158.yaml 原文)
OFFICIAL = dict(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=20,
)

BASE_SEED = 2026
NULL_SEEDS = [2026, 0, 1, 2, 3]  # base 含在内, 共 5 个同配置模型
PERTURBS = {
    "lr0.15": dict(learning_rate=0.15),
    "leaves180": dict(num_leaves=180),
    "l1_300": dict(lambda_l1=300.0),
    "col0.75": dict(colsample_bytree=0.75),
}
SHALLOW = dict(num_leaves=8, max_depth=3)  # 已知弱模型

Z_ALPHA, Z_POWER = 1.6449, 0.8416  # 单侧 α=0.05, power=0.8


def purge_end(cal, start, end, n=EMBARGO_DAYS):
    """段末截去 n 个交易日 (splits.yaml embargo.implementation)."""
    days = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    assert len(days) > n + 10, f"段 {start}~{end} 交易日不足"
    return days[0].strftime("%Y-%m-%d"), days[-1 - n].strftime("%Y-%m-%d")


def build_dataset():
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    cal = D.calendar(start_time="2008-01-01", end_time="2015-12-31")
    seg = {k: purge_end(cal, *v) for k, v in E1.items()}
    print(f"[seg] purged(embargo={EMBARGO_DAYS}): {seg}", flush=True)

    handler = Alpha158(
        instruments=MARKET,
        start_time=seg["train"][0],
        end_time=seg["promo"][1],
        fit_start_time=seg["train"][0],   # 预处理统计量只在 train 段拟合 (协议 preprocessing.rule)
        fit_end_time=seg["train"][1],
    )
    dataset = DatasetH(
        handler,
        segments={"train": seg["train"], "valid": seg["dev"], "test": seg["promo"]},
    )
    return dataset, seg


def daily_rankic(model, dataset):
    pred = model.predict(dataset, segment="test")
    label = dataset.prepare("test", col_set="label")
    both = pd.concat([pred.rename("score"), label.iloc[:, 0].rename("label")], axis=1).dropna()
    return both.groupby(level="datetime").apply(
        lambda d: d["score"].corr(d["label"], method="spearman")
    )


def train_one(name, params, dataset):
    from qlib.contrib.model.gbdt import LGBModel

    t0 = time.time()
    model = LGBModel(**params)
    model.fit(dataset)
    ic = daily_rankic(model, dataset)
    best_iter = getattr(getattr(model, "model", None), "best_iteration", None)
    ic.to_csv(OUT_IC / f"{name}.csv", header=["rank_ic"])
    print(
        f"[model] {name}: {time.time() - t0:.0f}s, best_iter={best_iter}, "
        f"promo RankIC mean={ic.mean():.4f} std={ic.std():.4f} n={ic.notna().sum()}",
        flush=True,
    )
    return ic


def pair_stats(kind, name_a, name_b, ic_a, ic_b):
    """a=parent/base, b=child/variant; d_t = ic_b - ic_a."""
    both = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    d = (both["b"] - both["a"]).to_numpy()
    n = len(d)
    mean, std = float(np.mean(d)), float(np.std(d, ddof=1))
    rho = float(np.corrcoef(d[:-1], d[1:])[0, 1]) if n > 2 else np.nan
    n_eff = n * (1 - rho) / (1 + rho) if np.isfinite(rho) and abs(rho) < 1 else n
    n_eff = float(min(max(n_eff, 1.0), n * 2))  # AR(1) 近似; 负自相关时上限 2N 防夸大
    se = std / np.sqrt(n_eff)
    # 配对 vs 独立样本设计: 独立设计的差分方差 = var(a)+var(b)
    var_unpaired = float(both["a"].var(ddof=1) + both["b"].var(ddof=1))
    return dict(
        kind=kind,
        pair=f"{name_b} - {name_a}",
        n_days=n,
        mean_d=round(mean, 5),
        std_d=round(std, 5),
        rho1=round(rho, 3),
        n_eff=round(n_eff, 1),
        t_eff=round(mean / se, 2),
        delta_min=round((Z_ALPHA + Z_POWER) * se, 5),
        corr_ab=round(float(both["a"].corr(both["b"])), 3),
        var_reduction_vs_unpaired=round(std**2 / var_unpaired, 3),
    )


def main():
    t0 = time.time()
    dataset, seg = build_dataset()
    print(f"[dataset] built in {time.time() - t0:.0f}s", flush=True)

    ics = {}
    for s in NULL_SEEDS:
        ics[f"seed{s}"] = train_one(f"seed{s}", {**OFFICIAL, "seed": s}, dataset)
    for name, delta in PERTURBS.items():
        ics[name] = train_one(name, {**OFFICIAL, **delta, "seed": BASE_SEED}, dataset)
    ics["shallow"] = train_one("shallow", {**OFFICIAL, **SHALLOW, "seed": BASE_SEED}, dataset)

    base = f"seed{BASE_SEED}"
    rows = []
    seeds = [f"seed{s}" for s in NULL_SEEDS]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            rows.append(pair_stats("NULL", seeds[i], seeds[j], ics[seeds[i]], ics[seeds[j]]))
    for name in PERTURBS:
        rows.append(pair_stats("PERTURB", base, name, ics[base], ics[name]))
    rows.append(pair_stats("GAP", base, "shallow", ics[base], ics["shallow"]))

    df = pd.DataFrame(rows)
    null_df = df[df["kind"] == "NULL"]
    summary = dict(
        episode="E1 (development)",
        promo_segment=list(seg["promo"]),
        n_promo_days=int(rows[0]["n_days"]),
        null_median_std_d=float(null_df["std_d"].median()),
        null_median_rho1=float(null_df["rho1"].median()),
        null_median_delta_min=float(null_df["delta_min"].median()),
        null_max_abs_t=float(null_df["t_eff"].abs().max()),
        null_false_pos_1sided_5pct=int((null_df["t_eff"].abs() > Z_ALPHA).sum()),
        base_promo_rankic_mean=float(ics[base].mean()),
        go_threshold=0.010,
        nogo_threshold=0.015,
    )
    med = summary["null_median_delta_min"]
    summary["verdict"] = "GO" if med <= 0.010 else ("NO-GO" if med > 0.015 else "MARGINAL")

    with open(HERE / "results.json", "w") as f:
        json.dump({"summary": summary, "pairs": rows, "segments": seg}, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200)
    print("\n===== pair table =====")
    print(df.to_string(index=False))
    print("\n===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
