"""Exp 2 — 序贯累积 (路线 A) × 组合层面评价量 (路线 B).

动机: Exp 1 测得单次选择的 winner's curse = 0.0036, 受物理上界 E[max_K]·sd_noise 锁死
      (充分训练模型的 sd_noise ≈ 0.002-0.003, 见 noise_probe/). 结论不是"没有现象",
      而是**测错了地方** —— 论文讲的是演化循环里的验收 (方案 §6.1 的方法线含
      "greedy+multi-seed"/"no-gate evolution", 8/03 定, 早于任何结果), 而 Exp 1 的
      proposer 是**无记忆随机的单次选择** (已在 exp1_phenomenon/README §6.1 记为
      "本文测得的是下界"). 本脚本把两个偏差同时修掉:

  路线 A (序贯): 贪心链 —— 每轮从当前 incumbent 变异出 K 个候选, 在**同一个 dev 段**上
    取 argmax 作为下一轮父代, 跑 T 轮. dev 段被复用 T 次 = Dwork 自适应数据分析的标准设定,
    退化随查询次数增长, **不像 E[max_K] 那样按 √(2lnK) 饱和**.
  路线 B (评价量): 同一批模型同时用两种量打分 ——
    ic     日频截面 RankIC 均值 (Exp 1 口径)
    sharpe 多空十分组组合的年化夏普 (官方四分项 Rank_SR 口径; DSR/PBO 文献正是为它而生)
    Sharpe 只由头尾两组决定, 排序微扰会让个股进出分组 → 预期 noise_share 更高.

判据 (对齐 §7-W2 的 0.010): 序贯 selection(T) 在任一评价量上 ≥0.010 且未饱和 → 现象成立.

协议与 Exp 1 **逐条一致** (复用 run_exp1.prepare_arrays): 冻结 splits v1 的 E1,
train 2008-2013 / dev 2014 / promo 2015, embargo=2, 预处理只在 train 段拟合,
特征未归一化(同 Exp 1, 保证与 0.0036 可比). **final_test(2016) 不触碰.**

运行: python run_exp2.py --chains 5 --rounds 25 --k 10 --single 128 --workers 24
产出: single_pool.csv, chains.csv, exp2_curve.csv, results.json
"""

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp1_phenomenon"))
from run_exp1 import (  # noqa: E402  复用同一份数据管线与预算, 保证与 Exp 1 可比
    OFFICIAL, NUM_BOOST_ROUND, EARLY_STOPPING, PREFILTER, anchor, prepare_arrays,
)

N_GROUP = 10          # 十分组多空 (官方 Rank_SR 口径)
ANN = np.sqrt(252.0)
METRICS = ("ic", "sharpe")


# ---------------------------------------------------------------- 评价量

def daily_scores(pred, tag, D):
    """一次前向给出两个日度序列: 截面 RankIC, 与多空十分组日收益."""
    from scipy.stats import rankdata

    b, labr, lab = D[f"bounds_{tag}"], D[f"labrank_{tag}"], D[f"lab_{tag}"]
    n = len(b) - 1
    ic, ls = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        s = slice(b[i], b[i + 1])
        p, lr, y = pred[s], labr[s], lab[s]
        m = len(p)
        if m < 3 or p.std() == 0:
            continue
        pr = rankdata(p)
        if lr.std() > 0:
            ic[i] = np.corrcoef(pr, lr)[0, 1]
        if m >= 2 * N_GROUP:  # 分组太小则该日无多空收益
            # 按预测分位切 10 组, 多头 = 最高组, 空头 = 最低组 (等权)
            q = (pr - 0.5) / m * N_GROUP
            top, bot = q >= N_GROUP - 1, q < 1
            if top.any() and bot.any():
                ls[i] = y[top].mean() - y[bot].mean()
    return ic, ls


def summarize(ic, ls):
    """日度序列 → 两个标量分数. sharpe 用年化多空夏普."""
    out = {"ic": float(np.nanmean(ic))}
    v = ls[~np.isnan(ls)]
    out["sharpe"] = float(v.mean() / v.std(ddof=1) * ANN) if len(v) > 2 and v.std(ddof=1) > 0 else np.nan
    return out


# ---------------------------------------------------------------- 变异

def mutate(parent, rng):
    """从 parent 出发的小扰动 —— 与 Exp 1 的 sample_narrow 同尺度, 但中心是 incumbent
    而非官方配置. 这正是"贪心链"与"从固定池里挑"的区别所在."""
    p = dict(parent)
    p["learning_rate"] = float(np.clip(p["learning_rate"] * np.exp(rng.normal(0, 0.20)), 0.05, 0.4))
    p["num_leaves"] = int(np.clip(p["num_leaves"] * np.exp(rng.normal(0, 0.25)), 32, 512))
    p["max_depth"] = int(np.clip(p["max_depth"] + rng.integers(-2, 3), 4, 12))
    p["colsample_bytree"] = float(np.clip(p["colsample_bytree"] + rng.normal(0, 0.06), 0.5, 1.0))
    p["subsample"] = float(np.clip(p["subsample"] + rng.normal(0, 0.06), 0.5, 1.0))
    p["lambda_l1"] = float(np.clip(p["lambda_l1"] * np.exp(rng.normal(0, 0.30)), 20, 800))
    p["lambda_l2"] = float(np.clip(p["lambda_l2"] * np.exp(rng.normal(0, 0.30)), 50, 2000))
    p["min_data_in_leaf"] = int(np.clip(p["min_data_in_leaf"] * np.exp(rng.normal(0, 0.4)), 5, 200))
    p["bagging_freq"] = 1
    p["seed"] = int(rng.integers(0, 10 ** 6))
    return p


# ---------------------------------------------------------------- 训练

_D = None
_THREADS = 4


def _init(d, threads):
    global _D, _THREADS
    _D, _THREADS = d, threads


def train_one(job):
    import lightgbm as lgb

    key, params = job
    t0 = time.time()
    p = {"objective": "mse", "verbosity": -1, "num_threads": _THREADS, "deterministic": True,
         **PREFILTER, **params}
    dtr = lgb.Dataset(_D["x_train_l"], label=_D["y_train_l"], free_raw_data=False)
    dva = lgb.Dataset(_D["x_dev_l"], label=_D["y_dev_l"], free_raw_data=False)
    bst = lgb.train(p, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dtr, dva],
                    valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    rec = {"key": key, "secs": round(time.time() - t0, 1),
           "best_iter": int(bst.best_iteration), "params": params}
    for tag in ("dev", "promo"):
        ic, ls = daily_scores(bst.predict(_D[f"x_{tag}_i"]), tag, _D)
        for m, v in summarize(ic, ls).items():
            rec[f"{tag}_{m}"] = v
    return rec


def run_batch(pool, jobs):
    return list(pool.imap_unordered(train_one, jobs, chunksize=1))


# ---------------------------------------------------------------- 曲线

def single_shot_curve(tab, metric, k_grid, n_resample=2000, seed=7):
    """Exp 1 的单次选择口径, 但可换评价量 —— 路线 B 的直接对照."""
    dev = tab[f"dev_{metric}"].to_numpy()
    promo = tab[f"promo_{metric}"].to_numpy()
    ok = np.isfinite(dev) & np.isfinite(promo)
    dev, promo = dev[ok], promo[ok]
    n = len(dev)
    drift = dev.mean() - promo.mean()
    rng = np.random.default_rng(seed)
    rows = []
    for K in k_grid:
        if K > n:
            continue
        sd, sp = np.empty(n_resample), np.empty(n_resample)
        for r in range(n_resample):
            idx = rng.choice(n, size=K, replace=False)
            w = idx[np.argmax(dev[idx])]
            sd[r], sp[r] = dev[w], promo[w]
        rows.append(dict(mode="single", metric=metric, x=K,
                         dev_sel=float(sd.mean()), promo_sel=float(sp.mean()),
                         optimism=float((sd - sp).mean()),
                         selection=float((sd - sp).mean() - drift),
                         se=float((sd - sp).std(ddof=1) / np.sqrt(n_resample))))
    return rows, dict(mode="single", metric=metric, n=n, drift=float(drift),
                      sd_dev=float(dev.std(ddof=1)), sd_promo=float(promo.std(ddof=1)),
                      mean_dev=float(dev.mean()), mean_promo=float(promo.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--single", type=int, default=128)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    D = prepare_arrays()
    print(f"[dataset] ready in {time.time() - t0:.0f}s", flush=True)
    ctx = get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_init, initargs=(D, args.threads))

    # ---------- 单次选择池 (路线 B: 同一批模型, 两种评价量) ----------
    rng = np.random.default_rng(20260812)
    a = anchor()
    cands = [a] + [mutate(a, rng) for _ in range(args.single - 1)]
    t1 = time.time()
    res = run_batch(pool, [(i, c) for i, c in enumerate(cands)])
    res.sort(key=lambda r: r["key"])
    single = pd.DataFrame([{k: r[k] for k in
                            ("key", "secs", "best_iter", "dev_ic", "promo_ic", "dev_sharpe", "promo_sharpe")}
                           for r in res])
    single.to_csv(HERE / "single_pool.csv", index=False)
    print(f"[single] {len(single)} 候选 {time.time() - t1:.0f}s | "
          f"dev_ic {single.dev_ic.min():.4f}~{single.dev_ic.max():.4f} | "
          f"dev_sharpe {single.dev_sharpe.min():.2f}~{single.dev_sharpe.max():.2f}", flush=True)

    curve_rows, diags = [], []
    for m in METRICS:
        r, d = single_shot_curve(single, m, [1, 2, 5, 10, 25, 50, 100])
        curve_rows += r
        diags.append(d)
        print(f"  [single/{m}] selection(K=100) = {r[-1]['selection']:.5f}", flush=True)

    # ---------- 序贯贪心链 (路线 A) ----------
    # 每轮: 每条链各生成 K 个变异 -> 全部并行训练 -> 各链在 **dev** 上取 argmax 作新 incumbent.
    # 选择用 dev (agent 可见), promo 只记录不参与选择 —— 密封纪律.
    chains = {c: dict(incumbent=dict(a), rng=np.random.default_rng(1000 + c)) for c in range(args.chains)}
    hist, all_eval = [], []
    t2 = time.time()
    for m in METRICS:  # 两种评价量各跑一组独立的链 (选择准则不同 → 轨迹不同)
        for c in chains:
            chains[c] = dict(incumbent=dict(a), rng=np.random.default_rng(1000 + c))
        # 记录 incumbent 起点
        base = run_batch(pool, [(("base", m, c), a) for c in chains])[0]
        for c in chains:
            hist.append(dict(metric=m, chain=c, round=0,
                             dev=base[f"dev_{m}"], promo=base[f"promo_{m}"]))
        for t in range(1, args.rounds + 1):
            jobs = []
            for c, st in chains.items():
                for j in range(args.k):
                    jobs.append(((m, c, t, j), mutate(st["incumbent"], st["rng"])))
            out = run_batch(pool, jobs)
            all_eval += [{"metric": m, "chain": k[1], "round": k[2],
                          "dev": r[f"dev_{m}"], "promo": r[f"promo_{m}"]}
                         for r in out for k in [r["key"]]]
            for c, st in chains.items():
                mine = [r for r in out if r["key"][1] == c]
                dv = np.array([r[f"dev_{m}"] for r in mine], dtype=float)
                last = [h for h in hist if h["metric"] == m and h["chain"] == c][-1]
                if not np.isfinite(dv).any():
                    hist.append(dict(metric=m, chain=c, round=t, dev=last["dev"], promo=last["promo"]))
                    continue
                best = mine[int(np.nanargmax(dv))]
                # 贪心验收: 只有 dev 分超过当前 incumbent 才接受 (这正是被 gate 取代的那条规则)
                if best[f"dev_{m}"] > last["dev"]:
                    st["incumbent"] = best["params"]
                    hist.append(dict(metric=m, chain=c, round=t,
                                     dev=best[f"dev_{m}"], promo=best[f"promo_{m}"]))
                else:
                    hist.append(dict(metric=m, chain=c, round=t, dev=last["dev"], promo=last["promo"]))
            if t % 5 == 0:
                el = time.time() - t2
                print(f"  [chain/{m}] round {t}/{args.rounds}, {el:.0f}s", flush=True)
    pool.close()
    pool.join()

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(HERE / "chains.csv", index=False)
    ev = pd.DataFrame(all_eval)

    # 序贯曲线: selection(t) = [dev(inc_t) - promo(inc_t)] - drift
    # drift 用**所有被评估候选**的 dev-promo 均差, 与轮次无关 (同 Exp 1 的分解口径)
    for m in METRICS:
        e = ev[ev.metric == m]
        drift = float((e["dev"] - e["promo"]).mean())
        h = hist_df[hist_df.metric == m]
        g = h.groupby("round")
        for t, sub in g:
            opt = (sub["dev"] - sub["promo"]).to_numpy(dtype=float)
            curve_rows.append(dict(mode="chain", metric=m, x=int(t),
                                   dev_sel=float(sub["dev"].mean()), promo_sel=float(sub["promo"].mean()),
                                   optimism=float(np.nanmean(opt)),
                                   selection=float(np.nanmean(opt) - drift),
                                   se=float(np.nanstd(opt, ddof=1) / np.sqrt(len(opt))) if len(opt) > 1 else np.nan))
        diags.append(dict(mode="chain", metric=m, n=int(len(e)), drift=drift,
                          n_chains=args.chains, n_rounds=args.rounds, k=args.k))

    curve = pd.DataFrame(curve_rows)
    # 标准化效应量: IC (~0.07) 与 Sharpe (~7.0) 量纲差 100 倍, 绝对值不可比.
    # 除以各自池的 sd_dev → "选择性乐观相当于几个候选间标准差", 跨评价量可比,
    # 也是与 §7-W2 的 0.010 门槛对话的正确方式 (门槛是 IC 单位, 见 §5 换算).
    sd_map = {d["metric"]: d.get("sd_dev") for d in diags if d.get("mode") == "single"}
    curve["selection_std"] = [
        (r.selection / sd_map[r.metric]) if sd_map.get(r.metric) else np.nan
        for r in curve.itertuples()
    ]
    curve.to_csv(HERE / "exp2_curve.csv", index=False)

    summary = {"episode": "E1 (development)", "segments": D["seg"],
               "wall_secs": round(time.time() - t0),
               "n_trainings": int(args.single + len(METRICS) * (args.chains + args.rounds * args.chains * args.k)),
               "note": "sharpe = 未计成本的多空十分组年化夏普, 作为选择统计量使用, 非可交易性声明"}
    for m in METRICS:
        # ⚠️ 必须用 curve["mode"] —— curve.mode 会解析成 DataFrame.mode() 方法, 过滤静默失效
        s = curve[(curve["mode"] == "single") & (curve["metric"] == m)]
        c = curve[(curve["mode"] == "chain") & (curve["metric"] == m)].sort_values("x")
        s_last = float(s.iloc[-1]["selection"]) if len(s) else np.nan
        s_last_std = float(s.iloc[-1]["selection_std"]) if len(s) else np.nan
        c_last = float(c.iloc[-1]["selection"]) if len(c) else np.nan
        c_last_std = float(c.iloc[-1]["selection_std"]) if len(c) else np.nan
        c_half = float(c.iloc[len(c) // 2]["selection"]) if len(c) else np.nan
        summary[m] = dict(
            single_maxK=s_last, single_maxK_std=s_last_std,
            chain_final=c_last, chain_final_std=c_last_std, chain_half=c_half,
            # 关键判据: 序贯是否**不饱和** (后半程仍在长) —— 这才是与单次选择的本质差异
            still_growing=bool(np.isfinite(c_last) and np.isfinite(c_half) and c_last > c_half * 1.15),
            seq_vs_single=(float(c_last / s_last) if np.isfinite(c_last) and np.isfinite(s_last)
                           and abs(s_last) > 1e-9 else None),
        )
    summary["verdict_ic_0.010"] = ("GO" if np.isfinite(summary["ic"]["chain_final"])
                                   and summary["ic"]["chain_final"] >= 0.010 else "BELOW-0.010")
    with open(HERE / "results.json", "w") as f:
        json.dump({"summary": summary, "diagnostics": diags,
                   "curve": curve.to_dict("records")}, f, indent=2, ensure_ascii=False, default=str)

    pd.set_option("display.width", 220)
    print("\n===== curve =====")
    print(curve.to_string(index=False))
    print("\n===== summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
