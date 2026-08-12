# Protocol A：Qlib 官方 LightGBM/Alpha158 基线复现记录

复现日期：2026-08-03。运行方式：`qrun` + 官方 YAML **原文未改**
（`workspace/project/qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`，
qlib @ `79633dd`）。唯一环境适配：`MLFLOW_ALLOW_FILE_STORE=true`（mlflow 新版要求，不影响计算）。

## 对表结果（官方 README「Results on CSI300 / Alpha158」表 vs 本次复现）

| 指标 | 官方 README | 本次复现 | 偏差 |
|---|---:|---:|---:|
| IC | 0.0448 | 0.0468 | +0.0020 |
| ICIR | 0.3660 | 0.3816 | +0.0156 |
| Rank IC | 0.0469 | 0.0490 | +0.0021 |
| Rank ICIR | 0.3877 | 0.4067 | +0.0190 |
| 年化超额收益（含成本） | 0.0901 | 0.0807 | −0.0094 |
| 信息比率（含成本） | 1.0164 | 0.9145 | −0.1019 |
| 最大回撤 | −0.1038 | −0.0861 | 浅 0.0177 |

**结论：复现成功。** 信号类指标偏差 ~4–5%（略好于官方），回测类指标偏差 ~10%（略差），
排序与量级完全一致。偏差来源（均为已知的正常因素，非实现错误）：
1. **数据快照年代不同**：社区 cn_data 多次重新 dump，与 README 数字生成时的快照不逐位一致
   （README FAQ 自己也说明数据更新会改变结果）；
2. **LightGBM 版本**：本环境 4.7.0，README 时代为 3.x，直方图与默认行为有差异；
3. `num_threads=20` 下 LightGBM 的非确定性（微小）。

复现值（而非 README 值）自此作为**本项目的 incumbent 基线参照**：
同一环境、同一数据快照下的自我对照才有意义。

## 运行细节

- 总耗时 ~1 分钟（数据加载 30s + 预处理 3s + 训练/预测/回测 ~25s）——
  单候选全流程成本极低，远低于计划的 10–15 分钟红线。
- 产物：`protocol_a/mlruns/258144598711872771/`（pred.pkl、port_analysis_1day.pkl 等）。
- 复跑命令（cwd = `examples/evoquant_qlib/protocol_a/`）：

```bash
MLFLOW_ALLOW_FILE_STORE=true conda run -n quant --no-capture-output \
  qrun "/root/quant/qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml"
```

## 官方协议要点（Protocol A 定义，供后文引用）

- market csi300，benchmark SH000300；
- Alpha158：start 2008-01-01，end 2020-08-01，fit 段 = train 段；
- 切分：train 2008-2014 / valid 2015-2016 / test 2017-01-01~2020-08-01；
- LGB 超参：lr 0.2、num_leaves 210、max_depth 8、λ1 205.6999、λ2 580.9768、
  colsample 0.8879、subsample 0.8789、loss mse；
- 回测：TopkDropoutStrategy topk=50 n_drop=5，open_cost 5bp / close_cost 15bp / min_cost 5，
  limit_threshold 0.095，deal_price close。

> 注意：Protocol A 只用于"实现正确性锚点"。论文主实验走 Protocol B
> （四段切分 + 滚动 episode），切分冻结后另行记录。
