# factor_evolution — 用 LLM 从原始行情演化选股因子

主循环（`python -m famou`）+ 真实 qlib 数据的因子挖掘实验。LLM 反复改写 `compute_factor`，
评估器在冻结的 CSI300 快照上算 RankIC，高分的被选来继续变异。

## 先填 API key

打开 `config.yaml`，找到 `infrastructure.llm`，把三个 `FILL_ME_IN` 换掉：

```yaml
infrastructure:
  llm:
    provider: openai            # 兼容 OpenAI 协议的端点都填 openai
    api_base: https://...       # ← 你的
    model: deepseek-v3          # ← 你的
    api_key: sk-...             # ← 你的
```

两个坑：

- **`config/key_set.yaml` 是死文件**，全仓库没有代码读它，往那写不生效。
- **不支持环境变量展开**，`api_key: ${OPENAI_API_KEY}` 会被原样当成 key 发出去。必须写字面量。

## 跑

```bash
cd famou-quant
/opt/conda/envs/quant/bin/python -m famou \
  -c examples/evoquant_qlib/factor_evolution/config.yaml \
  -p examples/evoquant_qlib/factor_evolution/seed_factor.py \
  -e examples/evoquant_qlib/factor_evolution/evaluator.py
```

必须用 `quant` 环境（qlib 0.9.7 装在那儿）。`config.yaml` 里 `max_iterations` 默认 3，
先跑通链路再往上加。

产出在 `famou_data/<experiment_id>/`，可以用监控面板实时看：

```bash
/opt/conda/envs/quant/bin/python -m famou.monitor --run famou_data/ --open
```

## 不需要 key 的自检

```bash
# 评估器全套测试（含前视检测）
/opt/conda/envs/quant/bin/python -m pytest examples/evoquant_qlib/factor_evolution/tests/ -v

# 单独给种子因子打分
/opt/conda/envs/quant/bin/python examples/evoquant_qlib/factor_evolution/evaluator.py
```

## 数据与口径

默认 **E1（development episode）**。所有日期、universe、label、embargo 都从
`../protocol_b/splits_v2.yaml` 直接读取，本目录不抄一个日期——抄了就会和冻结协议悄悄漂移。

| | 区间 | 用途 |
|---|---|---|
| train | 2008-01-01 ~ 2013-12-31 | **不评分**，只让滚动窗口在评分区间首日就预热 |
| visible_dev | 2014-01-01 ~ 2014-12-31 | 打分（243 个交易日，245 减去 2 天 embargo） |

universe `csi300`（按时点成分，无幸存者偏差），标签 `Ref($close,-2)/Ref($close,-1)-1`。

### 为什么默认 E1

E1 的 final_test 在 v1 时期已被反复查看 → `splits_v2.yaml:75` 判为**作废**。
作废反而让它有用：它是**可以随便糟蹋的沙盒**。方法还在动的时候（种子长什么样、
prompt 怎么写、fitness 用哪个、跑多少轮），就该在这里看，因为看它不要钱。

E2–E11 是 evaluation episode，它们唯一的价值就是**还没人看过**。方法定稿、只跑不改之后再用：

```bash
FAMOU_EPISODE=E11 /opt/conda/envs/quant/bin/python -m famou -c ... -p ... -e ...
```

E11 额外是**唯一的 post-cutoff episode**（评估段晚于 GPT-4o 2024-06 与 DeepSeek-R1 2023-12
的知识截止日），对"LLM 是不是在背已知因子"最有说服力，也因此最不该拿来调参。

> 本实验只读 `train + visible_dev`。任何 episode 的 `sealed_promotion` 和 `final_test`
> 都不会被这套代码碰到。

每条评估结果里都带 `episode` 字段——不同 episode 的分数**不可比**（年份不同、regime 不同）。

## 指标

`combined_score` = **dev 段日度 RankIC 均值**，这是驱动演化的量。其余都进 metrics：

| 字段 | 含义 |
|---|---|
| `rank_ic` | 日度 Spearman(因子, 未来收益) 均值 = `combined_score` |
| `ic` | Pearson 版本 |
| `icir` | `rank_ic / std(日度 RankIC)` |
| `sharpe` | 多空十分位组合年化夏普 |
| `long_short_return` | 该组合年化收益 |
| `subperiod_rank_ic` | 评分区间切 4 段各自的 RankIC（看是否只在某段有效） |
| `n_ic_days` / `coverage` | 有效天数 / 因子覆盖率 |
| `episode` | 这条分数来自哪个 episode（不同 episode 不可比） |

Sharpe 用的是**未归一化**的远期收益（`panel.fwd_ret_raw`），不是协议里那个
CSZScoreNorm 过的 label——z-score 之后的"收益"没有量纲，拿它算夏普会得到一个
看着合理但没有意义的数。RankIC 不受这个变换影响，所以两个标签各用各的。

## 防作弊：前视检测

给 LLM 原始 OHLCV，就一定会有候选去读未来数据，而这种因子 RankIC 能到 **1.0000、
夏普 51**（实测），会瞬间统治搜索并伪装成重大发现。所以做的是**因果性检验**而不是人工审代码：

把面板截断到第 T 天重新调用 `compute_factor`，第 T 行必须与完整运行**逐位相同**，
NaN 的位置也要一致。用了未来数据不可能通过。失败的候选 `validity=0`、得分 0，
`error_info` 里写明是哪一天、差多少，debug 循环可以据此让 LLM 自己修。

测试覆盖了四种前视：直接用未来收益、差一位切片、`np.nanmean(axis=0)` 这类全样本统计量
（最容易无意写出来的一种）、以及把泄漏藏在 NaN 分布里。

另外，候选代码跑在**子进程**里且只允许产出因子矩阵，**所有指标都在父进程算**——
候选没法自己上报分数。`tests/test_evaluator.py` 里有一条专门测这个。

## 文件

| 文件 | 作用 |
|---|---|
| `config.yaml` | 实验配置 + **API key 填这里** + 给 LLM 的任务说明 |
| `seed_factor.py` | 种子因子（20 日反转）与 `compute_factor` 契约 |
| `evaluator.py` | 三道关：前视 → 健全性 → 打分；`EPISODE` 默认值在这里 |
| `panel.py` | 原始 OHLCV 面板构建与磁盘缓存 |

面板首次构建约 10s，之后走 `~/.cache/famou_panels` 缓存约 0.1s；单个候选评估约 4s。
每个 episode 各有自己的缓存。

## 基准

种子因子（−20 日收益，真·反转）在 **E1 dev（2014）** 上：

```
rank_ic  +0.0184    icir  +0.092    sharpe  -0.90    n_ic_days 243
subperiod_rank_ic  [+0.029, +0.042, +0.049, -0.047]
```

三点值得注意：

1. **RankIC 为正但 Sharpe 为负。** 秩相关方向对，但驱动多空组合的**两端**方向反了——
   说明极值区的股票表现与整体秩关系相悖。这是个真实信号：别只看 RankIC。
2. **第 4 段翻负**（前三段 +0.03~+0.05，第四段 −0.047）。因子在 2014 年后期失效。
3. 同一个种子在 E11（2024）上是 `rank_ic -0.0039`——**短期反转在 2014 有效、2024 无效**。
   这也是为什么两个 episode 的分数不能放在一起比。

### 关于 `reliability/baseline_formula.py` 的一个发现

那份 formula 基线用 `-1.0 * ROC20` 并在 docstring 里称之为 "20-day reversal"，
`run_reliability_real.py` 的注释也说它是 "a plain 20-day reversal score"。但 qlib 的
Alpha158 定义是 `ROC%d = Ref($close, d)/$close`，即 **close[t−20]/close[t]**，是普通收益率的**倒数**。
因此 `-1.0*ROC20` 关于收益率**单调递增**——它其实是**动量**，不是反转。

实测（同一面板、同一天）：本目录种子 `-(r)` 与 `-1.0*ROC20` 的截面 Spearman = **−1.0000**，
RankIC 恰好相反（+0.0184 / −0.0184）。两者是同一效应的相反方向，互相印证了实现的正确性。

影响：reliability 那条线的 formula incumbent 既**名实不符**，又因为 IC 为负而**翻个符号就能赢**，
削弱了"打败 incumbent 说明找到了教科书之外的东西"这个论证。要修的话把权重改成 `+1.0` 即可。
