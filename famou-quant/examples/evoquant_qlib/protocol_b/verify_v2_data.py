"""Protocol v2 数据验证 — 新快照能否支撑 post-cutoff episode.

检查四件事 (任一失败则 v2 不成立):
  1. qlib 能在新 provider 上初始化, 日历覆盖到 2026;
  2. Alpha158 能构建, 且 **VWAP0 不再整列 NaN** (v1 快照缺 vwap 字段, 见 exp1 README §5);
  3. csi300 成分在 post-cutoff 区间 (2025-2026) 有效;
  4. 各 episode 分段的交易日数足够 (dev/promo ≥200, final_test 允许末端截断).

运行: python verify_v2_data.py
"""
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd

NEW = "/root/.qlib/qlib_data/cn_data_20260810"
# 主流 LLM 知识截止日 (DeepFund 2505.11065 点名): DeepSeek-R1 2023-12, GPT-4o 2024-06
CUTOFF = pd.Timestamp("2024-06-30")

EPISODES = {  # train 6y / dev 1y / promo 1y / final_test 1y, 滚动 1y
    f"E{i+1}": dict(train=(f"{2008+i}-01-01", f"{2013+i}-12-31"),
                    visible_dev=(f"{2014+i}-01-01", f"{2014+i}-12-31"),
                    sealed_promotion=(f"{2015+i}-01-01", f"{2015+i}-12-31"),
                    final_test=(f"{2016+i}-01-01", f"{2016+i}-12-31"))
    for i in range(11)
}


def main():
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=NEW, region="cn")
    cal = D.calendar(start_time="2000-01-01", end_time="2026-12-31")
    print(f"[1] 日历 {cal[0].date()} ~ {cal[-1].date()}, {len(cal)} 交易日")

    # 2. Alpha158 + vwap
    from qlib.contrib.data.handler import Alpha158
    h = Alpha158(instruments="csi300", start_time="2025-01-01", end_time="2026-06-30",
                 fit_start_time="2025-01-01", fit_end_time="2025-12-31")
    df = h.fetch()
    nan_all = df.columns[df.isna().all()].tolist()
    print(f"[2] Alpha158 on 2025-2026: {df.shape}, 全 NaN 列 = {nan_all or '无'}")
    if "VWAP0" in df.columns:
        print(f"    VWAP0 缺失率 = {df['VWAP0'].isna().mean():.4f} (v1 快照为 1.0000)")

    # 3. 成分股
    for probe in ("2019-06-03", "2025-06-03", "2026-06-03"):
        inst = D.list_instruments(D.instruments("csi300"), start_time=probe, end_time=probe, as_list=True)
        print(f"[3] csi300 @ {probe}: {len(inst)} 只")

    # 4. 分段交易日数 + post-cutoff 判定
    print(f"[4] episode 分段 (交易日数; ✅=评估段完全晚于 {CUTOFF.date()})")
    rows = []
    for name, segs in EPISODES.items():
        cnt, ok = {}, True
        for k, (s, e) in segs.items():
            n = len([d for d in cal if pd.Timestamp(s) <= d <= pd.Timestamp(e)])
            cnt[k] = n
            if k in ("visible_dev", "sealed_promotion") and n < 200:
                ok = False
        post = pd.Timestamp(segs["sealed_promotion"][0]) > CUTOFF
        usable = ok and cnt["final_test"] >= 100
        rows.append(dict(episode=name, **cnt, post_cutoff=post, usable=usable))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    pc = t[t.post_cutoff & t.usable]["episode"].tolist()
    print(f"\n可用的 post-cutoff episode: {pc or '无'}")


if __name__ == "__main__":
    main()
