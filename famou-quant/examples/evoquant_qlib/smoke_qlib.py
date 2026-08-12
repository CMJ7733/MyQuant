"""Smoke test: 验证本机 qlib 调用链路端到端可用.

四步递进, 每步独立 PASS/FAIL, 任一步失败后续跳过:
  1. qlib.init + 交易日历
  2. CSI300 股票池 + 原始行情 D.features
  3. Alpha158 handler 构建小窗口数据集
  4. LightGBM 快速训练 + 日度 RankIC

只验证链路, 不评价分数. 运行:
  conda run -n quant --no-capture-output python smoke_qlib.py
"""

import os
import sys
import time
import traceback

# qlib 的 LGBModel.fit 会经 R.log_metrics 走 mlflow 文件后端;
# mlflow 新版默认禁用文件后端, 不设此变量会在 fit 末尾抛异常.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PROVIDER_URI = "/root/.qlib/qlib_data/cn_data"
MARKET = "csi300"
BENCHMARK = "SH000300"

# 小窗口: 只为快速验证链路. 正式协议的四段切分另行冻结, 与此无关.
TRAIN = ("2015-01-01", "2016-12-31")
VALID = ("2017-01-01", "2017-06-30")
TEST = ("2017-07-01", "2017-12-31")


def step(name):
    def deco(fn):
        fn._step_name = name
        return fn
    return deco


@step("1. qlib.init + calendar")
def step_init(ctx):
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    cal = D.calendar(start_time="2015-01-01", end_time="2020-12-31")
    assert len(cal) > 1200, f"交易日数量异常: {len(cal)}"
    ctx["calendar_len"] = len(cal)
    return f"{len(cal)} trading days (2015-2020), first={cal[0].date()}, last={cal[-1].date()}"


@step("2. instruments + D.features")
def step_features(ctx):
    from qlib.data import D

    instruments = D.instruments(market=MARKET)
    stocks = D.list_instruments(instruments, start_time=TEST[0], end_time=TEST[1], as_list=True)
    assert len(stocks) > 250, f"CSI300 成分数量异常: {len(stocks)}"

    df = D.features(
        stocks[:5],
        ["$close", "$volume", "Ref($close, 1)", "$close/Ref($close, 1) - 1"],
        start_time=TEST[0],
        end_time=TEST[1],
    )
    assert not df.empty and df["$close"].notna().any(), "行情为空"
    ctx["n_stocks"] = len(stocks)
    return f"{len(stocks)} instruments in {MARKET}; features shape={df.shape}"


@step("3. Alpha158 dataset")
def step_alpha158(ctx):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments=MARKET,
        start_time=TRAIN[0],
        end_time=TEST[1],
        fit_start_time=TRAIN[0],
        fit_end_time=TRAIN[1],
    )
    dataset = DatasetH(
        handler,
        segments={"train": TRAIN, "valid": VALID, "test": TEST},
    )
    df_train = dataset.prepare("train", col_set=["feature", "label"])
    assert df_train.shape[0] > 50_000, f"训练样本异常: {df_train.shape}"
    assert df_train["feature"].shape[1] == 158, f"特征数≠158: {df_train['feature'].shape[1]}"
    ctx["dataset"] = dataset
    return f"train shape={df_train.shape} (features={df_train['feature'].shape[1]})"


@step("4. LightGBM + daily RankIC")
def step_lgb_rankic(ctx):
    import pandas as pd
    from qlib.contrib.model.gbdt import LGBModel

    model = LGBModel(
        loss="mse",
        num_boost_round=50,
        early_stopping_rounds=20,
        num_leaves=64,
        learning_rate=0.1,
        num_threads=4,
    )
    model.fit(ctx["dataset"])
    pred = model.predict(ctx["dataset"], segment="test")  # Series, index=(datetime, instrument)

    label = ctx["dataset"].prepare("test", col_set="label")
    both = pd.concat([pred.rename("score"), label.iloc[:, 0].rename("label")], axis=1).dropna()
    daily_rank_ic = both.groupby(level="datetime").apply(
        lambda d: d["score"].corr(d["label"], method="spearman")
    )
    assert daily_rank_ic.notna().sum() > 60, f"有效 IC 天数不足: {daily_rank_ic.notna().sum()}"
    mean_ic, std_ic = daily_rank_ic.mean(), daily_rank_ic.std()
    return (
        f"test days={daily_rank_ic.notna().sum()}, "
        f"RankIC mean={mean_ic:.4f}, std={std_ic:.4f}, ICIR={mean_ic / std_ic:.3f} "
        f"(链路验证, 数值不作评价)"
    )


def main():
    steps = [step_init, step_features, step_alpha158, step_lgb_rankic]
    ctx, failed = {}, False
    for fn in steps:
        name = fn._step_name
        if failed:
            print(f"[SKIP] {name}")
            continue
        t0 = time.time()
        try:
            msg = fn(ctx)
            print(f"[PASS] {name} ({time.time() - t0:.1f}s) — {msg}")
        except Exception:
            failed = True
            print(f"[FAIL] {name} ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    print("\nRESULT:", "FAIL" if failed else "ALL PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
