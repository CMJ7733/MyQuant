# evoquant_qlib — EvoQuant 论文实验的 Qlib example

ICLR 2027 投稿《EvoQuant: Reliability-Gated Self-Evolution of Stock Prediction Algorithms》
的实验目录。设计文档见 `workspace/project/docs/iclr2027_evoquant_plan_review.md`。

> **与 BigAlpha 竞赛严格隔离**：本目录只使用 Qlib 公开数据，
> 不得引用竞赛数据、竞赛平台产物或其派生指标（保密义务长期有效）。

## 仓库分工（2026-08-03 起）

| 仓库 | 用途 | 分支纪律 |
|---|---|---|
| `/root/quant/F4Q/famou-quant`（本仓库，clone 自 famou-v2） | **论文 / qlib 实验专用**。框架改造（gate/certified-archive 插件等）只发生在这里 | 工作分支 `evoquant-paper` |
| `/root/quant/baidu/acg-fm/famou-v2` | **BigAlpha 比赛专用**，保持 master（`0209c2c`）不动 | 不合并论文分支；比赛期间不改框架 |

clone 时未跟踪的比赛目录（`quant_e2e`/`quant_factor`/`fm-job`）天然未被带入本仓库——
两边零交叉。本仓库 origin 指向 famou-v2，可在其切回 master 后 `git push origin evoquant-paper` 作备份。

## 冻结的环境（论文可复现性要求，改动须记录在此）

| 项 | 值 | 冻结日期 |
|---|---|---|
| conda env | `quant` (Python 3.12.13) | 2026-08-03 |
| qlib | 0.9.8.dev32，本地源码安装自 `workspace/project/qlib` @ commit `79633dd9506ea689e5400dea0197717b5b3d74b7` | 2026-08-03 |
| 数据 | qlib 社区 cn_data 快照 → `/root/.qlib/qlib_data/cn_data`，日历 1999-11-10 ~ **2020-09-25**（详见 `data_provenance.txt`） | 2026-08-03 |

## 当前状态

- [x] 目录创建
- [x] `smoke_qlib.py` 全部通过（2026-08-03：4/4 PASS，Alpha158 构建 20.6s、LGB 训练+RankIC 3.9s，
      小窗口 test RankIC≈0.050/ICIR≈0.55，仅作链路验证）
- [x] Protocol A：官方 LightGBM/Alpha158 基线复现（2026-08-03：官方 YAML 原文 qrun，
      IC 0.0468 vs 官方 0.0448，偏差已解释，详见 `protocol_a/README.md`）
- [x] 四段数据协议（Train / Visible-Dev / Sealed-Promotion / Final-Test）：
      `protocol_b/splits.yaml` 冻结 v1（2026-08-03，embargo=2 交易日=标签前视深度，
      E1 为 development episode，E2–E5 为评估 episode；
      **冻结 commit：`08b83a515e1e837470412078edaaeb4bdbeac41d`**，分支 `evoquant-paper`）
- [x] Power study（W1 Go/No-Go #1）：**GO**（2026-08-04：E1 promo 228 天，
      NULL 对中位 std(d_t)=0.051、Δ_min=0.0091≤0.010；配对方差缩减 13–20×；
      seed 噪声地板 0.003–0.007；gate 设计参数已冻结，详见 `power_study/README.md`）
- [x] Exp 1 现象曲线（W2 Go/No-Go #2）：**字面 NO-GO，但机制已定量**
      （2026-08-11：narrow 池 n=128，selection(K=100)=**0.0036**，门槛 0.010；
      K=50 即封顶——上界 `E[max_K]·sd_noise`=0.0044 物理不可达 0.010。
      噪声地板 sd_dev=0.00177；标度律 `selection≈E[max_K]·sd_noise²/sd_dev`
      corr=0.90/0.94。达标需 sd_noise≳0.003 → **下一步测 NN 候选类的 sd_noise**。
      详见 `exp1_phenomenon/README.md`）
- [ ] visible evaluator + sealed gate 服务
- [ ] famou strategy 接入（certified archive + promotion gate）

## 运行

```bash
conda run -n quant --no-capture-output python smoke_qlib.py
```

`smoke_qlib.py` 分四步，逐步打印 PASS/FAIL，全过则 qlib 调用链路可用：
1. `qlib.init` + 交易日历读取；
2. CSI300 成分与个股原始行情 `D.features`；
3. Alpha158 handler 小窗口构建数据集；
4. LightGBM 快速训练 + 测试段日度 RankIC（只验证链路，不看分数好坏）。
