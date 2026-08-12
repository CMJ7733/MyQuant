"""Exp 1 — adaptive overfitting 现象曲线 (W2 Go/No-Go 门 #2).

问题: 一个 greedy agent 在可见 dev 段上从 K 个候选里挑 argmax, 被挑中者在密封 promo 段上
      的表现比它的 dev 分数低多少? 这个 optimism 随 K 增长吗? 量级够不够写成论文现象?

Go/No-Go 判据 (评审文档 §7-W2): dev-promo gap 曲线可见、量级 ≥0.01 → GO.

设计:
  - 协议: 冻结 splits v1 的 **E1** (development episode). train=2008-2013 / dev=2014 /
    promo=2015, embargo=2 尾修剪, 预处理只在 train 段拟合 —— 与 power_study 逐条一致.
    **final_test(2016) 全程不触碰.**
  - 训练管线: 复刻 qlib LGBModel.fit 的语义 (train/valid 用 DK_L, 预测用 DK_I,
    early_stopping=50, num_boost_round=1000), 但直接调 lgb.train 以跳过 mlflow 逐轮日志.
  - 两个候选池 (同一管线, 两种提案分布):
      broad  — 9 维超参空间上均匀/对数均匀随机采样, 模拟无先验的 naive proposer
      narrow — 官方配置邻域内的小扰动, 模拟已收敛到好区域的后期演化搜索 (对论文更关键)
    每池含官方配置本体作为锚点 (candidate 0).
  - 现象曲线: 对每个 K, 从池中无放回抽 K 个候选 R 次, 取 dev 均值 RankIC 最大者:
      optimism(K)   = IC_dev(sel) - IC_promo(sel)                     总乐观度
      selection(K)  = optimism(K) - (mean_dev_pool - mean_promo_pool)  扣掉 regime 漂移的选择性乐观
      transfer(K)   = IC_promo(sel) - mean_promo_pool                  选择是否真带来样本外收益
    baseline 漂移项与 K 无关, 故 selection(K) 才是 winner's curse 本身.

诚实边界: 本脚本的 proposer 是**无记忆随机**的, 候选间不共享反馈. 真实 LLM greedy agent
  逐轮依赖 dev 反馈提案, 其 adaptive overfitting 只会更强 —— 故此处测得的是**下界**.

运行: /opt/conda/envs/quant/bin/python run_exp1.py [--n-broad 160 --n-narrow 160 --workers 16]
产出: pool_{broad,narrow}/candidates.csv, ic_dev.csv, ic_promo.csv; curve.csv; results.json
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
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent

PROVIDER_URI = "/root/.qlib/qlib_data/cn_data"
MARKET = "csi300"
EMBARGO_DAYS = 2  # = label.forward_depth_days (protocol_b/splits.yaml)

# E1 名义边界 (splits.yaml 冻结 v1). final_test(2016) 不出现在此脚本任何位置.
E1 = {
    "train": ("2008-01-01", "2013-12-31"),
    "dev": ("2014-01-01", "2014-12-31"),
    "promo": ("2015-01-01", "2015-12-31"),
}

# 官方 LightGBM/Alpha158 超参 (workflow_config_lightgbm_Alpha158.yaml 原文)
OFFICIAL = dict(
    learning_rate=0.2,
    num_leaves=210,
    max_depth=8,
    colsample_bytree=0.8879,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    min_data_in_leaf=20,  # lightgbm 默认; 官方 yaml 未指定
)
NUM_BOOST_ROUND, EARLY_STOPPING = 300, 50
# 单候选算力预算 (评审文档 §6.3: "单候选训练 ≤10-15 分钟"). 官方配置在 lr=0.2 下 15-30 轮即早停,
# 300 轮上限只对低 learning_rate 候选生效 —— 命中率记入 candidates.csv 的 hit_cap, 须在 README 披露.
# 不设上限时 lr≈0.05 + num_leaves=512 的候选实测单个可跑 >19 分钟, 整池耗时不可预测.
LGB_THREADS_DEFAULT = 4  # 实测 t=20 比 t=8 更慢 → 单模型已受内存带宽限制, 并行度放在候选级

# 全池统一关闭 feature_pre_filter. 理由: 开启时 lightgbm 会按**每个候选自己的**
# min_data_in_leaf 预剔除不可分裂特征, 使分箱结果与候选绑定 → 无法跨候选复用已分箱 Dataset,
# 而重复分箱正是吞吐瓶颈. 实测开/关会改变树结构 (best_iter 22→44), 故必须**对所有候选
# 一致关闭**(含 anchor), 使池内可比; 代价是 anchor 的绝对 IC 与 power_study 略有差异.
# 这是有意的协议选择, 不是实现疏漏 —— 论文披露.
PREFILTER = {"feature_pre_filter": False}

PARAM_KEYS = list(OFFICIAL) + ["bagging_freq", "seed"]
K_GRID = [1, 2, 5, 10, 25, 50, 100]
N_RESAMPLE = 2000


# ---------------------------------------------------------------- 数据

def purge_end(cal, start, end, n=EMBARGO_DAYS):
    """段末截去 n 个交易日 (splits.yaml embargo.implementation)."""
    days = [d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    assert len(days) > n + 10, f"段 {start}~{end} 交易日不足"
    return days[0].strftime("%Y-%m-%d"), days[-1 - n].strftime("%Y-%m-%d")


def prepare_arrays():
    """建 Alpha158 数据集并一次性物化成 numpy, 供所有候选复用 (fork 后 COW 共享)."""
    import qlib
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    cal = D.calendar(start_time="2008-01-01", end_time="2015-12-31")
    seg = {k: purge_end(cal, *v) for k, v in E1.items()}
    print(f"[seg] purged(embargo={EMBARGO_DAYS}): {seg}", flush=True)

    handler = Alpha158(
        instruments=MARKET,
        start_time=seg["train"][0],
        end_time=seg["promo"][1],
        fit_start_time=seg["train"][0],  # 预处理统计量只在 train 段拟合 (协议 preprocessing.rule)
        fit_end_time=seg["train"][1],
    )
    dataset = DatasetH(
        handler,
        segments={"train": seg["train"], "valid": seg["dev"], "test": seg["promo"]},
    )

    out = {"seg": seg}
    # 训练 / 早停: DK_L (含 DropnaLabel + 标签截面 z-score), 与 LGBModel._prepare_data 一致
    for key, tag in (("train", "train"), ("valid", "dev")):
        df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        out[f"x_{tag}_l"] = np.ascontiguousarray(df["feature"].values, dtype=np.float64)
        out[f"y_{tag}_l"] = np.ascontiguousarray(np.squeeze(df["label"].values), dtype=np.float64)
    # 评估: DK_I (与 LGBModel.predict 一致); RankIC 对标签的逐日单调变换不敏感,
    # 故 DK_I 原始收益标签与 DK_L 的截面 z-score 标签给出同一 spearman 值
    # 注意: 特征侧**不做 dropna** —— lightgbm 原生处理 NaN, 且本快照缺 vwap 字段导致
    # Alpha158 的 VWAP0 整列为 NaN, 按行 dropna 会清空整个数据集. 只要求标签非 NaN
    # (= power_study.daily_rankic 的行为: pred 恒非 NaN, dropna 只落在 label 上).
    for key, tag in (("valid", "dev"), ("test", "promo")):
        x = dataset.prepare(key, col_set="feature", data_key=DataHandlerLP.DK_I)
        y = dataset.prepare(key, col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
        keep = y.notna().to_numpy()
        x, y = x[keep], y[keep]
        order = np.argsort(x.index.get_level_values("datetime").to_numpy(), kind="stable")
        x, y = x.iloc[order], y.iloc[order]
        out[f"x_{tag}_i"] = np.ascontiguousarray(x.values, dtype=np.float64)
        out[f"lab_{tag}"] = np.ascontiguousarray(y.values, dtype=np.float64)
        days = x.index.get_level_values("datetime")
        out[f"days_{tag}"] = np.asarray(sorted(days.unique()))
        # 逐日切片边界 (已按 datetime 排序 → 连续块)
        codes = pd.factorize(days, sort=True)[0]
        out[f"bounds_{tag}"] = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
        # 标签的逐日截面秩, 预算一次
        lab_rank = np.empty_like(out[f"lab_{tag}"])
        b = out[f"bounds_{tag}"]
        for i in range(len(b) - 1):
            lab_rank[b[i]:b[i + 1]] = rankdata(out[f"lab_{tag}"][b[i]:b[i + 1]])
        out[f"labrank_{tag}"] = lab_rank
    for tag in ("train", "dev"):
        print(f"[data] {tag}_l: {out[f'x_{tag}_l'].shape}", flush=True)
    for tag in ("dev", "promo"):
        print(f"[data] {tag}_i: {out[f'x_{tag}_i'].shape}, "
              f"{len(out[f'days_{tag}'])} 交易日", flush=True)
    return out


def daily_rank_ic(pred, tag, D):
    """逐日截面 spearman(pred, label) —— 对已排序数组按日切片算 rank 的 pearson."""
    b, labr = D[f"bounds_{tag}"], D[f"labrank_{tag}"]
    ic = np.empty(len(b) - 1)
    for i in range(len(b) - 1):
        s = slice(b[i], b[i + 1])
        pr, lr = rankdata(pred[s]), labr[s]
        if len(pr) < 3 or pr.std() == 0 or lr.std() == 0:
            ic[i] = np.nan
        else:
            ic[i] = np.corrcoef(pr, lr)[0, 1]
    return ic


# ---------------------------------------------------------------- 候选池

def sample_broad(rng):
    """9 维超参空间随机采样 —— 无先验 proposer."""
    subsample = float(rng.uniform(0.5, 1.0))
    return dict(
        learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        num_leaves=int(np.exp(rng.uniform(np.log(16), np.log(512)))),
        max_depth=int(rng.choice([4, 6, 8, 10, 12])),
        colsample_bytree=float(rng.uniform(0.5, 1.0)),
        subsample=subsample,
        lambda_l1=float(np.exp(rng.uniform(np.log(1.0), np.log(500.0)))),
        lambda_l2=float(np.exp(rng.uniform(np.log(1.0), np.log(1000.0)))),
        min_data_in_leaf=int(np.exp(rng.uniform(np.log(10), np.log(500)))),
        bagging_freq=1,  # 官方 yaml 漏了它 → 那里的 subsample 其实是空转; 此处让它真正生效
        seed=int(rng.integers(0, 10**6)),
    )


def sample_narrow(rng):
    """官方配置邻域内的小扰动 —— 模拟已收敛的后期演化搜索."""
    p = dict(OFFICIAL)
    p["learning_rate"] = float(np.clip(OFFICIAL["learning_rate"] * np.exp(rng.normal(0, 0.20)), 0.05, 0.4))
    p["num_leaves"] = int(np.clip(OFFICIAL["num_leaves"] * np.exp(rng.normal(0, 0.25)), 32, 512))
    p["max_depth"] = int(np.clip(OFFICIAL["max_depth"] + rng.integers(-2, 3), 4, 12))
    p["colsample_bytree"] = float(np.clip(OFFICIAL["colsample_bytree"] + rng.normal(0, 0.06), 0.5, 1.0))
    p["subsample"] = float(np.clip(OFFICIAL["subsample"] + rng.normal(0, 0.06), 0.5, 1.0))
    p["lambda_l1"] = float(np.clip(OFFICIAL["lambda_l1"] * np.exp(rng.normal(0, 0.30)), 20, 800))
    p["lambda_l2"] = float(np.clip(OFFICIAL["lambda_l2"] * np.exp(rng.normal(0, 0.30)), 50, 2000))
    p["min_data_in_leaf"] = int(np.clip(OFFICIAL["min_data_in_leaf"] * np.exp(rng.normal(0, 0.4)), 5, 200))
    p["bagging_freq"] = 1
    p["seed"] = int(rng.integers(0, 10**6))
    return p


def anchor():
    """官方配置本体 (bagging_freq=0 → subsample 空转, 与官方 yaml 行为一致)."""
    return {**OFFICIAL, "bagging_freq": 0, "seed": 2026}


def sample_replicate(rng):
    """只换 seed 的官方配置 —— 直接测同一配置的纯重训噪声 (W1 噪声地板的 dev/promo 双段版).

    用途: 候选池的 sd_dev 里有多少是真实质量差异、多少是噪声, 决定了 winner's curse 的量级.
    sd_dev² ≈ sd_true² + sd_noise², 本池给出 sd_noise.
    """
    return {**OFFICIAL, "bagging_freq": 0, "seed": int(rng.integers(0, 10**6))}


# ---------------------------------------------------------------- 训练

_D = None  # fork 继承的数据
_THREADS = LGB_THREADS_DEFAULT
_DS = {}   # 每 worker 一份已分箱 Dataset, 跨候选复用


def _init(d, threads):
    """每个 worker 只分箱一次 —— 32 个 worker 各自重复分箱 406840x158 是本实验的吞吐瓶颈
    (实测单候选 7s 独占 → 266s 并发). 复用要求 feature_pre_filter=False, 见 PREFILTER 说明.
    """
    import lightgbm as lgb

    global _D, _THREADS
    _D, _THREADS = d, threads
    _DS["train"] = lgb.Dataset(d["x_train_l"], label=d["y_train_l"], free_raw_data=False,
                               params=PREFILTER).construct()
    _DS["valid"] = lgb.Dataset(d["x_dev_l"], label=d["y_dev_l"], free_raw_data=False,
                               params=PREFILTER, reference=_DS["train"]).construct()


def train_one(job):
    import lightgbm as lgb

    idx, params = job
    t0 = time.time()
    # deterministic=True: 同 seed + 同 num_threads 可复现 (实测 replicate 池两次独立运行
    # 的 sd/range/spearman 逐位一致).
    p = {"objective": "mse", "verbosity": -1, "num_threads": _THREADS, "deterministic": True,
         **PREFILTER, **params}
    booster = lgb.train(
        p, _DS["train"], num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[_DS["train"], _DS["valid"]], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    ic_dev = daily_rank_ic(booster.predict(_D["x_dev_i"]), "dev", _D)
    ic_promo = daily_rank_ic(booster.predict(_D["x_promo_i"]), "promo", _D)
    return dict(
        cand=idx, best_iter=int(booster.best_iteration),
        hit_cap=int(booster.best_iteration >= NUM_BOOST_ROUND),
        secs=round(time.time() - t0, 1), ic_dev=ic_dev, ic_promo=ic_promo, **params,
    )


def run_pool(name, sampler, n, seed, workers, threads, D):
    rng = np.random.default_rng(seed)
    cands = [anchor()] + [sampler(rng) for _ in range(n - 1)]
    jobs = list(enumerate(cands))
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(D, threads)) as pool:
        res = []
        for r in pool.imap_unordered(train_one, jobs, chunksize=1):
            res.append(r)
            if len(res) % 25 == 0:
                el = time.time() - t0
                print(f"  [{name}] {len(res)}/{n} done, {el:.0f}s "
                      f"(eta {el / len(res) * (n - len(res)):.0f}s)", flush=True)
    res.sort(key=lambda r: r["cand"])

    out = HERE / f"pool_{name}"
    out.mkdir(exist_ok=True)
    ic_dev = pd.DataFrame({f"c{r['cand']}": r["ic_dev"] for r in res}, index=D["days_dev"])
    ic_promo = pd.DataFrame({f"c{r['cand']}": r["ic_promo"] for r in res}, index=D["days_promo"])
    ic_dev.to_csv(out / "ic_dev.csv")
    ic_promo.to_csv(out / "ic_promo.csv")
    cols = ["cand", "best_iter", "hit_cap", "secs"] + PARAM_KEYS
    tab = pd.DataFrame([{k: r[k] for k in cols} for r in res])
    tab["dev_ic"] = ic_dev.mean().to_numpy()
    tab["promo_ic"] = ic_promo.mean().to_numpy()
    tab.to_csv(out / "candidates.csv", index=False)
    print(f"[{name}] {n} 候选 {time.time() - t0:.0f}s; hit_cap={int(tab.hit_cap.sum())}/{n}; "
          f"secs p50={tab.secs.median():.0f} p95={tab.secs.quantile(.95):.0f}; "
          f"dev_ic {tab.dev_ic.min():.4f}~{tab.dev_ic.max():.4f}, "
          f"promo_ic {tab.promo_ic.min():.4f}~{tab.promo_ic.max():.4f}", flush=True)
    return tab, ic_dev, ic_promo


# ---------------------------------------------------------------- 现象曲线

def phenomenon_curve(name, tab, rng_seed=7):
    """对每个 K 抽 R 次 K 元子集, 取 dev argmax, 记其 dev/promo 表现."""
    dev, promo = tab["dev_ic"].to_numpy(), tab["promo_ic"].to_numpy()
    n = len(dev)
    mean_dev, mean_promo = float(dev.mean()), float(promo.mean())
    drift = mean_dev - mean_promo  # regime 漂移, 与 K 无关
    rng = np.random.default_rng(rng_seed)
    rows = []
    for K in K_GRID:
        if K > n:
            continue
        sel_dev, sel_promo = np.empty(N_RESAMPLE), np.empty(N_RESAMPLE)
        for r in range(N_RESAMPLE):
            idx = rng.choice(n, size=K, replace=False)
            w = idx[np.argmax(dev[idx])]
            sel_dev[r], sel_promo[r] = dev[w], promo[w]
        opt = sel_dev - sel_promo
        rows.append(dict(
            pool=name, K=K,
            dev_sel=round(float(sel_dev.mean()), 5),
            promo_sel=round(float(sel_promo.mean()), 5),
            optimism=round(float(opt.mean()), 5),
            optimism_se=round(float(opt.std(ddof=1) / np.sqrt(N_RESAMPLE)), 5),
            selection=round(float(opt.mean() - drift), 5),   # winner's curse 本身
            transfer=round(float(sel_promo.mean() - mean_promo), 5),
            promo_sel_p10=round(float(np.percentile(sel_promo, 10)), 5),
            frac_promo_below_mean=round(float((sel_promo < mean_promo).mean()), 3),
        ))
    curve = pd.DataFrame(rows)
    spear = float(pd.Series(dev).corr(pd.Series(promo), method="spearman"))
    diag = dict(
        pool=name, n_cand=n, mean_dev=round(mean_dev, 5), mean_promo=round(mean_promo, 5),
        drift_dev_minus_promo=round(drift, 5),
        sd_dev=round(float(dev.std(ddof=1)), 5), sd_promo=round(float(promo.std(ddof=1)), 5),
        spearman_dev_promo=round(spear, 3),
        anchor_dev=round(float(dev[0]), 5), anchor_promo=round(float(promo[0]), 5),
        best_dev_cand=int(np.argmax(dev)), best_promo_cand=int(np.argmax(promo)),
    )
    return curve, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-broad", type=int, default=200)
    ap.add_argument("--n-narrow", type=int, default=300)
    ap.add_argument("--n-replicate", type=int, default=24)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=LGB_THREADS_DEFAULT)
    args = ap.parse_args()

    t0 = time.time()
    D = prepare_arrays()
    print(f"[dataset] ready in {time.time() - t0:.0f}s", flush=True)

    # 噪声地板: 同配置只换 seed, 给出 sd_noise (dev/promo 两段各一)
    rep, _, _ = run_pool("replicate", sample_replicate, args.n_replicate, 20260810,
                         args.workers, args.threads, D)
    noise = dict(
        n=int(len(rep)),
        sd_dev=round(float(rep["dev_ic"].std(ddof=1)), 5),
        sd_promo=round(float(rep["promo_ic"].std(ddof=1)), 5),
        spearman_dev_promo=round(float(rep["dev_ic"].corr(rep["promo_ic"], method="spearman")), 3),
        range_dev=[round(float(rep["dev_ic"].min()), 5), round(float(rep["dev_ic"].max()), 5)],
        range_promo=[round(float(rep["promo_ic"].min()), 5), round(float(rep["promo_ic"].max()), 5)],
    )
    print(f"[noise floor] {noise}", flush=True)

    curves, diags = [], []
    for name, sampler, n, seed in (
        ("narrow", sample_narrow, args.n_narrow, 20260811),
        ("broad", sample_broad, args.n_broad, 20260812),
    ):
        tab, _, _ = run_pool(name, sampler, n, seed, args.workers, args.threads, D)
        c, d = phenomenon_curve(name, tab)
        # 噪声占比: dev 排序里有多少是真实质量差异. sd_true² = max(sd_dev² - sd_noise², 0)
        d["noise_share_dev"] = round(min(1.0, (noise["sd_dev"] / d["sd_dev"]) ** 2), 3)
        d["noise_share_promo"] = round(min(1.0, (noise["sd_promo"] / d["sd_promo"]) ** 2), 3)
        curves.append(c)
        diags.append(d)
        print(c.to_string(index=False), flush=True)

    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(HERE / "curve.csv", index=False)

    # 判据: 论文关心的是后期收敛搜索, 故以 narrow 池最大 K 的选择性乐观为准
    nar = curve[curve.pool == "narrow"]
    head = float(nar[nar.K == nar.K.max()]["selection"].iloc[0])
    verdict = "GO" if head >= 0.010 else ("NO-GO" if head < 0.005 else "MARGINAL")
    summary = dict(
        episode="E1 (development)", segments=D["seg"],
        n_days_dev=len(D["days_dev"]), n_days_promo=len(D["days_promo"]),
        k_grid=K_GRID, n_resample=N_RESAMPLE, noise_floor=noise,
        headline_pool="narrow", headline_K=int(nar.K.max()),
        headline_selection_optimism=round(head, 5),
        go_threshold=0.010, nogo_threshold=0.005, verdict=verdict,
        wall_secs=round(time.time() - t0),
    )
    with open(HERE / "results.json", "w") as f:
        json.dump({"summary": summary, "diagnostics": diags,
                   "curve": curve.to_dict("records")}, f, indent=2, ensure_ascii=False, default=str)

    print("\n===== diagnostics =====")
    print(pd.DataFrame(diags).to_string(index=False))
    print("\n===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
