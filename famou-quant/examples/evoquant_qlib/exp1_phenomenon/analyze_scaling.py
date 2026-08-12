"""Exp 1 附加分析 — winner's curse 的标度律 (dev 窗长扫描).

动机: 主曲线只回答"在 242 天 dev 窗、GBDT 候选类上, 选择性乐观有多大". 但 Go/No-Go 要的是
      "这个方向可不可行", 需要知道**什么条件下它变大**. 本脚本给出可外推的定量关系.

理论: 记 dev 分数 = 真实质量 + 噪声, 池内 sd_dev² = sd_true² + sd_noise².
      从 K 个候选取 dev argmax, 选择性乐观 (扣掉与 K 无关的 regime 漂移后) 近似为
          selection(K) ≈ E[max_K z] · sd_noise² / sd_dev
      即: 噪声占 dev 方差的比例越高、K 越大, winner's curse 越大; 上限是 E[max_K z]·sd_noise.
      而 sd_noise ∝ 1/√(dev 天数) —— 所以缩短 dev 窗会放大现象, 且是可预测地放大.

做法: 逐日 IC 已落盘, 故改变 dev 窗长**不需要重训任何模型** —— 只是对不同天数取均值.
      取 dev 段**末尾** W 天 (紧邻 promo, 与真实滚动部署一致).
      sd_noise(W) 由 replicate 池 (同配置只换 seed) 在同一窗上直接测得, 不靠假设.

运行: /opt/conda/envs/quant/bin/python analyze_scaling.py
产出: scaling.csv, scaling_report.md
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
K_GRID = [2, 5, 10, 25, 50, 100]
W_GRID = [20, 40, 60, 120, 242]
N_RESAMPLE = 4000


def emax(K, n=400000, rng=None):
    """E[max of K standard normals] —— 蒙特卡洛."""
    rng = rng or np.random.default_rng(0)
    return float(rng.standard_normal((n // K + 1, K)).max(axis=1).mean())


def load(pool):
    d = pd.read_csv(HERE / f"pool_{pool}" / "ic_dev.csv", index_col=0, parse_dates=True)
    p = pd.read_csv(HERE / f"pool_{pool}" / "ic_promo.csv", index_col=0, parse_dates=True)
    return d, p


def curve(dev, promo, rng):
    """给定每候选的 dev/promo 标量分数, 返回 selection(K)."""
    n = len(dev)
    drift = dev.mean() - promo.mean()
    out = {}
    for K in K_GRID:
        if K > n:
            continue
        sd, sp = np.empty(N_RESAMPLE), np.empty(N_RESAMPLE)
        for r in range(N_RESAMPLE):
            idx = rng.choice(n, size=K, replace=False)
            w = idx[np.argmax(dev[idx])]
            sd[r], sp[r] = dev[w], promo[w]
        out[K] = dict(
            selection=float((sd - sp).mean() - drift),
            transfer=float(sp.mean() - promo.mean()),
            se=float((sd - sp).std(ddof=1) / np.sqrt(N_RESAMPLE)),
        )
    return out


def main():
    emax_tab = {K: emax(K) for K in K_GRID}
    rows = []
    for pool in ("narrow", "broad"):
        if not (HERE / f"pool_{pool}" / "ic_dev.csv").exists():
            continue
        dev_d, promo_d = load(pool)
        rep_d, _ = load("replicate")
        promo_full = promo_d.mean().to_numpy()  # promo 始终用全窗 (它是被评估对象, 不是被搜索对象)
        for W in W_GRID:
            if W > len(dev_d):
                continue
            dev = dev_d.iloc[-W:].mean().to_numpy()          # dev 段末尾 W 天
            sd_noise = float(rep_d.iloc[-W:].mean().std(ddof=1))  # 同窗上的纯重训噪声
            sd_dev = float(dev.std(ddof=1))
            noise_share = min(1.0, (sd_noise / sd_dev) ** 2)
            rng = np.random.default_rng(11)
            c = curve(dev, promo_full, rng)
            for K, v in c.items():
                rows.append(dict(
                    pool=pool, W=W, K=K, n_cand=len(dev),
                    sd_dev=round(sd_dev, 5), sd_noise=round(sd_noise, 5),
                    noise_share=round(noise_share, 3),
                    selection=round(v["selection"], 5), se=round(v["se"], 5),
                    predicted=round(emax_tab[K] * sd_noise ** 2 / sd_dev, 5),
                    ceiling=round(emax_tab[K] * sd_noise, 5),
                    transfer=round(v["transfer"], 5),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "scaling.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))

    # 预测 vs 实测的一致性 (标度律是否成立)
    for pool in df.pool.unique():
        s = df[df.pool == pool]
        r = float(np.corrcoef(s.selection, s.predicted)[0, 1])
        ratio = float((s.selection / s.predicted.replace(0, np.nan)).median())
        print(f"\n[{pool}] selection vs predicted: corr={r:.3f}, median ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
