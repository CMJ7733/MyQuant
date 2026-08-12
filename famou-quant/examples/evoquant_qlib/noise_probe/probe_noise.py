"""噪声地板探针 — 跨候选类测 sd_noise (Exp 1 的直接后续).

背景: Exp 1 测得 GBDT 候选类的重训噪声 sd_noise(dev)=0.00177, 由此 winner's curse 的物理上界
      E[max_K]·sd_noise = 0.0044, 达不到 §7-W2 的 0.010 门槛. 反解出的达标条件是
      **sd_noise ≳ 0.003**. 本脚本在其他候选类上直接测这一个数.

判据: sd_noise(dev) ≥ 0.003 → 该候选类可支撑 Go/No-Go #2 达标.

候选类 (plan §6.3 的主力搜索空间 "LightGBM 变体 + 小型 MLP/GRU/TCN"):
  linear  qlib LinearModel(ols)        —— **负对照**, 无随机源, 预期 sd_noise ≈ 0
  mlp     qlib DNNModelPytorch         —— 随机初始化 + minibatch 洗牌 + 早停点抖动
  lstm    qlib pytorch_lstm_ts.LSTM    —— 同上 + 时序窗口 (TSDatasetH)
  gbdt    lightgbm (Exp 1 已测 0.00177, 此处仅供同脚本复核)

协议与 Exp 1 完全一致: 冻结 splits v1 的 E1, train 2008-2013 / dev 2014 / promo 2015,
embargo=2 尾修剪, 预处理只在 train 段拟合. **final_test(2016) 不触碰.**
只换 seed, 其余超参固定 —— 测的是纯重训噪声, 不是搜索空间的质量差异.

运行: python probe_noise.py --model mlp --seeds 16 --workers 16
产出: noise_<model>.json + 逐 seed 的 dev/promo 日度 IC csv
"""

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROVIDER_URI = os.environ.get("QLIB_DATA", "/root/.qlib/qlib_data/cn_data")
MARKET = "csi300"
EMBARGO_DAYS = 2
E1 = {
    "train": ("2008-01-01", "2013-12-31"),
    "dev": ("2014-01-01", "2014-12-31"),
    "promo": ("2015-01-01", "2015-12-31"),
}
STEP_LEN = 20  # lstm 时序窗口; 远小于协议上限 240 交易日


def purge_end(cal, start, end, n=EMBARGO_DAYS):
    days = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    assert len(days) > n + 10, f"段 {start}~{end} 交易日不足"
    return days[0].strftime("%Y-%m-%d"), days[-1 - n].strftime("%Y-%m-%d")


def build(model_kind):
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH, TSDatasetH

    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    cal = D.calendar(start_time="2008-01-01", end_time="2015-12-31")
    seg = {k: purge_end(cal, *v) for k, v in E1.items()}
    # ⚠️ Alpha158 的 infer_processors 默认是**空**的 —— Exp 1/power_study 用的是裸构造,
    # 即特征未归一化(对 LightGBM 无碍, 它对单调变换不敏感). 但 linear/MLP/LSTM 必须归一化,
    # 且本快照缺 vwap 字段导致 VWAP0 整列 NaN, 不 Fillna 会让 dropna 清空数据集.
    # 故本探针统一采用**官方 workflow_config_lightgbm_Alpha158.yaml 的处理器配置**,
    # 四个候选类完全一致 → 跨类可比. 代价: 本探针的 gbdt 数值与 Exp 1 的 0.00177 不直接可比,
    # 因此 gbdt 也在本管线下重测一次作为同口径基准.
    infer_processors = [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ]
    handler = Alpha158(
        instruments=MARKET, start_time=seg["train"][0], end_time=seg["promo"][1],
        fit_start_time=seg["train"][0], fit_end_time=seg["train"][1],
        infer_processors=infer_processors, learn_processors=learn_processors,
    )
    segments = {"train": seg["train"], "valid": seg["dev"], "test": seg["promo"]}
    if model_kind == "lstm":
        ds = TSDatasetH(handler=handler, segments=segments, step_len=STEP_LEN)
    else:
        ds = DatasetH(handler, segments=segments)
    return ds, seg


def make_model(kind, seed, n_feat=158, max_steps=None):
    if kind == "linear":
        from qlib.contrib.model.linear import LinearModel
        return LinearModel(estimator="ols")          # 无随机源 —— 负对照
    if kind == "gbdt":
        from qlib.contrib.model.gbdt import LGBModel
        return LGBModel(loss="mse", colsample_bytree=0.8879, learning_rate=0.2,
                        subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
                        max_depth=8, num_leaves=210, num_threads=4, seed=seed)
    if kind == "mlp":
        from qlib.contrib.model.pytorch_nn import DNNModelPytorch
        # ⚠️ max_steps 决定训练是否充分. 欠训练模型的重训噪声天然偏高, 拿它当结论=自欺,
        # 故两档都测并同时报告: 300(类默认, 约 1.5 epoch) 与 3000(接近收敛).
        return DNNModelPytorch(
            lr=0.001, max_steps=max_steps or 300, batch_size=2000,
            early_stop_rounds=50, eval_steps=20,
            optimizer="adam", loss="mse", GPU=-1, seed=seed,
            pt_model_kwargs={"input_dim": n_feat, "layers": (256,)},
        )
    if kind == "lstm":
        from qlib.contrib.model.pytorch_lstm_ts import LSTM
        return LSTM(d_feat=n_feat, hidden_size=64, num_layers=2, dropout=0.0,
                    n_epochs=max_steps or 30, lr=0.001, batch_size=2000, early_stop=10,
                    n_jobs=2, GPU=-1, seed=seed)
    raise ValueError(kind)


def daily_rank_ic(pred, dataset, segment):
    label = dataset.prepare(segment, col_set="label")
    if hasattr(label, "iloc") and label.ndim == 2:
        label = label.iloc[:, 0]
    both = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
    return both.groupby(level="datetime").apply(
        lambda d: d["score"].corr(d["label"], method="spearman") if len(d) > 2 else np.nan
    )


_G = {}


def _init(kind, max_steps=None, workers=1):
    # ⚠️ torch/OMP 默认每进程用满所有核 —— 24 worker x 128 线程 = 3100+ 线程, 实测 load 冲到 78,
    # 15 分钟连一个 seed 都跑不完. 必须按 worker 数均分. (Exp 1 的 lightgbm 走 num_threads 参数,
    # 没踩到这个坑; torch 没有等价的显式参数, 默认就是吃满.)
    n = max(1, (os.cpu_count() or 8) // max(1, workers))
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    _G["kind"] = kind
    _G["max_steps"] = max_steps
    _G["ds"], _G["seg"] = build(kind)


def run_seed(seed):
    kind = _G["kind"]
    ds = _G["ds"]
    t0 = time.time()
    model = make_model(kind, seed, max_steps=_G.get("max_steps"))
    model.fit(ds)
    ic_dev = daily_rank_ic(model.predict(ds, segment="valid"), ds, "valid")
    ic_promo = daily_rank_ic(model.predict(ds, segment="test"), ds, "test")
    dt = time.time() - t0
    print(f"  [{kind}] seed={seed} {dt:.0f}s dev={ic_dev.mean():.5f} promo={ic_promo.mean():.5f}",
          flush=True)
    return dict(seed=seed, secs=round(dt, 1), ic_dev=ic_dev, ic_promo=ic_promo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["linear", "mlp", "lstm", "gbdt"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    t0 = time.time()

    if args.workers <= 1:
        _init(args.model, args.max_steps, 1)
        res = [run_seed(s) for s in seeds]
    else:
        ctx = get_context("fork")
        with ctx.Pool(args.workers, initializer=_init, initargs=(args.model, args.max_steps, args.workers)) as pool:
            res = list(pool.imap_unordered(run_seed, seeds))
        res.sort(key=lambda r: r["seed"])

    dev = pd.DataFrame({f"s{r['seed']}": r["ic_dev"] for r in res})
    promo = pd.DataFrame({f"s{r['seed']}": r["ic_promo"] for r in res})
    dev.to_csv(out / f"ic_dev_{args.model}{args.tag}.csv")
    promo.to_csv(out / f"ic_promo_{args.model}{args.tag}.csv")

    m_dev, m_promo = dev.mean().to_numpy(), promo.mean().to_numpy()
    sd_dev = float(m_dev.std(ddof=1)) if len(m_dev) > 1 else 0.0
    sd_promo = float(m_promo.std(ddof=1)) if len(m_promo) > 1 else 0.0
    summary = dict(
        model=args.model, tag=args.tag, max_steps=args.max_steps, n_seeds=len(seeds), step_len=STEP_LEN if args.model == "lstm" else None,
        mean_dev=round(float(m_dev.mean()), 5), mean_promo=round(float(m_promo.mean()), 5),
        sd_noise_dev=round(sd_dev, 5), sd_noise_promo=round(sd_promo, 5),
        range_dev=[round(float(m_dev.min()), 5), round(float(m_dev.max()), 5)],
        spearman_dev_promo=(round(float(pd.Series(m_dev).corr(pd.Series(m_promo), method="spearman")), 3)
                            if len(m_dev) > 2 else None),
        # Exp 1 反解: K=100 达标需 sd_noise ≳0.003 (实测/理论比值 1.4)
        target_sd_noise=0.003,
        verdict=("PASS" if sd_dev >= 0.003 else "FAIL"),
        implied_selection_K100=round(1.4 * 2.51 * sd_dev, 5),  # 纯噪声池上界
        secs_per_seed_p50=float(np.median([r["secs"] for r in res])),
        wall_secs=round(time.time() - t0),
    )
    with open(out / f"noise_{args.model}{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
