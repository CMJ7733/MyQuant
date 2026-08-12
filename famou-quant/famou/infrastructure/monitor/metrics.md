# WandB 监控指标说明文档

## 概述

本文档详细说明了 Famou 2.0 在 WandB 监控看板中记录的所有指标。这些指标在每次迭代后自动上传，提供进化过程的实时可视化。

## 指标分类

监控指标分为以下几类：

1. **Iteration (迭代追踪)** - 实验进度
2. **Performance (性能指标)** - 分数统计
3. **Population (种群指标)** - 种群规模和结构
4. **Diversity (多样性指标)** - 种群多样性
5. **Cost (成本指标)** - LLM API 成本（如果启用）

---

## 1. Iteration (迭代追踪)

### `iteration/iteration`

**类型**: 整数
**范围**: 0, 1, 2, ..., max_iterations
**说明**: 当前迭代次数

- **用途**: X轴，追踪实验进度
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```
iteration/iteration: 0  → 初始状态
iteration/iteration: 1  → 第一次迭代完成
iteration/iteration: 10 → 第十次迭代完成
```

---

## 2. Performance (性能指标)

性能指标描述了种群中程序的分数分布情况。

### 2.1 `performance/best_score`

**类型**: 浮点数
**范围**: [0, ∞)，取决于具体任务
**说明**: 当前种群中的最佳分数

- **计算**: `max(scores)`
- **用途**: 追踪最佳性能随时间的改进
- **更新时机**: 每次迭代后（包括 iteration 0 初始状态）
- **图表类型**: 折线图（单调不递减）

**示例**:
```python
# Circle Packing 任务
performance/best_score: 0.8  → Iteration 0 初始最佳分数
performance/best_score: 1.2  → Iteration 5 改进
performance/best_score: 2.5  → Iteration 10 接近最优
```

### 2.2 `performance/avg_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 当前种群的平均分数

- **计算**: `mean(scores)`
- **用途**: 追踪整体性能水平
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
performance/avg_score: 0.65  → 平均水平
performance/avg_score: 0.85  → 整体提升
```

### 2.3 `performance/median_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 当前种群的中位数分数

- **计算**: `median(scores)`
- **用途**: 了解分数的中间值，不受极端值影响
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
performance/median_score: 0.68  → 中等程序的表现
```

### 2.4 `performance/std_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 分数的标准差

- **计算**: `std(scores)`
- **用途**: 衡量分数的离散程度
  - 高标准差: 种群中性能差异大
  - 低标准差: 种群趋于一致
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
performance/std_score: 0.15  → 较一致的种群
performance/std_score: 0.50  → 高度多样化的种群
```

### 2.5 `performance/min_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 当前种群中的最低分数

- **计算**: `min(scores)`
- **用途**: 追踪最差程序的性能
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

### 2.6 `performance/p10_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 第10百分位分数（10%的程序低于此分数）

- **计算**: `percentile(scores, 10)`
- **用途**: 了解低分段的性能
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

### 2.7 `performance/p90_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 第90百分位分数（90%的程序低于此分数）

- **计算**: `percentile(scores, 90)`
- **用途**: 了解高分段的性能
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

---

## 3. Population (种群指标)

种群指标描述了当前种群的规模和结构。

### 3.1 `population/size`

**类型**: 整数
**范围**: [1, ∞)
**说明**: 当前种群中的程序总数

- **计算**: `len(population)`
- **用途**: 追踪种群规模变化
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
population/size: 10  → 初始种群
population/size: 50  → 种群增长
population/size: 100 → 种群继续增长
```

### 3.2 `population/avg_generation`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 当前种群的平均代数

- **计算**: `mean([p.generation for p in population])`
- **用途**: 追踪进化的深度
  - 低代数: 新产生的程序较多
  - 高代数: 老程序占主导
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
population/avg_generation: 1.2  → 平均第1.2代
population/avg_generation: 5.8  → 平均第5.8代（深度演化）
```

### 3.3 `population/max_generation`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 当前种群中的最高代数

- **计算**: `max([p.generation for p in population])`
- **用途**: 了解最深演化代数
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

### 3.4 `population/min_generation`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 当前种群中的最低代数

- **计算**: `min([p.generation for p in population])`
- **用途**: 检查是否有初始程序仍在种群中
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

### 3.5 `population/unique_programs`

**类型**: 整数
**范围**: [1, population/size]
**说明**: 唯一代码的数量（去重）

- **计算**: `len(set([p.code for p in population]))`
- **用途**: 衡量代码多样性
  - 等于 population/size: 所有程序代码不同
  - 小于 population/size: 存在代码重复
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
population/size: 50
population/unique_programs: 50  → 所有程序唯一
population/unique_programs: 30  → 存在重复
```

### 3.6 `population/unique_ids`

**类型**: 整数
**范围**: [1, population/size]
**说明**: 唯一 ID 的数量

- **计算**: `len(set([p.id for p in population]))`
- **用途**: 检查 ID 唯一性（应该等于 population/size）
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

---

## 4. Diversity (多样性指标)

多样性指标评估种群的基因和代码多样性。

### 4.1 `diversity/genetic_diversity`

**类型**: 浮点数
**范围**: [0, 1]
**说明**: 基因多样性（不同世代的比例）

- **计算**: `unique_generations / total_programs`
- **用途**: 衡量世代多样性
  - 1.0: 每个程序来自不同世代
  - 0.5: 一半程序来自同一世代
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
# 种群有 10 个程序，来自 5 个不同世代
diversity/genetic_diversity: 0.5  # 5/10
```

### 4.2 `diversity/code_diversity`

**类型**: 浮点数
**范围**: [0, 1]
**说明**: 代码多样性（唯一代码的比例）

- **计算**: `unique_codes / total_programs`
- **用途**: 衡量代码级别的多样性
  - 1.0: 所有程序代码不同
  - < 1.0: 存在代码重复或相似
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
# 种群有 10 个程序，其中 8 个代码不同
diversity/code_diversity: 0.8  # 8/10
```

### 4.3 `diversity/parent_diversity`

**类型**: 浮点数
**范围**: [0, 1]
**说明**: 父代多样性（不同父代的比例）

- **计算**: `unique_parents / total_programs_with_parents`
- **用途**: 衡量谱系多样性
  - 1.0: 所有程序来自不同父代
  - < 1.0: 某些父代产生了多个后代
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

**示例**:
```python
# 8 个程序有父代，来自 4 个不同父代
diversity/parent_diversity: 0.5  # 4/8
```

### 4.4 `diversity/semantic_diversity` ✨

**类型**: 浮点数
**范围**: [0, 2]
**说明**: 语义多样性（基于特征向量的平均余弦距离）

- **计算**: 平均成对余弦距离
  - 对所有程序对计算余弦距离
  - 余弦距离 = 1 - 余弦相似度
  - 余弦相似度 = dot(vec1, vec2) / (||vec1|| × ||vec2||)
- **用途**: 衡量程序语义/行为层面的多样性
  - 0.0: 所有程序在语义上完全相同
  - 0.5-1.0: 中等多样性
  - 1.5-2.0: 高度多样性（程序语义差异很大）
- **更新时机**: 每次迭代后
- **图表类型**: 折线图
- **依赖**: 程序需要有 `feature_vector` 属性

**示例**:
```python
# 初始阶段：程序多样化
diversity/semantic_diversity: 1.2  # 高多样性

# 中期阶段：开始收敛
diversity/semantic_diversity: 0.8  # 中等多样性

# 后期阶段：收敛到优秀解
diversity/semantic_diversity: 0.3  # 低多样性（语义相似）
```

**与其他多样性指标的对比**:
- `genetic_diversity`: 衡量代数多样性（结构层面）
- `code_diversity`: 衡量代码文本多样性（语法层面）
- **`semantic_diversity`**: 衡量程序语义多样性（**行为层面**）⭐

**为什么需要语义多样性？**
- 两个程序代码可能不同，但行为相似
- 语义多样性能更好地反映程序功能层面的差异
- 对于 LLM 生成的代码尤其重要（不同 prompt 可能生成相似功能的代码）

---

## 5. Cost (成本指标)

成本指标追踪 LLM API 的使用情况（需要在 LLM 客户端中集成成本追踪）。

> **注意**: 这些指标需要在 LLM 客户端中记录 token 使用情况才能正常工作。

### 5.1 `cost/total_calls`

**类型**: 整数
**范围**: [0, ∞)
**说明**: LLM API 调用总次数

- **用途**: 追踪 API 调用频率
- **更新时机**: 每次 LLM 调用后（如果集成）
- **图表类型**: 折线图（单调递增）

### 5.2 `cost/input_tokens`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 输入 Token 总数

- **用途**: 追踪输入 Token 消耗
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

### 5.3 `cost/output_tokens`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 输出 Token 总数

- **用途**: 追踪输出 Token 消耗
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

### 5.4 `cost/total_tokens`

**类型**: 整数
**范围**: [0, ∞)
**说明**: Token 总数（输入 + 输出）

- **计算**: `input_tokens + output_tokens`
- **用途**: 追踪总 Token 消耗
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

### 5.5 `cost/input_cost_usd`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 输入 Token 成本（美元）

- **计算**: `input_tokens / 1,000,000 * cost_per_1M_input`
- **用途**: 追踪输入成本
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

**成本参考**（每百万 Token 价格）:
- GPT-4: $30 (input), $60 (output)
- GPT-4 Turbo: $10 (input), $30 (output)
- GPT-3.5 Turbo: $0.50 (input), $1.50 (output)

### 5.6 `cost/output_cost_usd`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 输出 Token 成本（美元）

- **计算**: `output_tokens / 1,000,000 * cost_per_1M_output`
- **用途**: 追踪输出成本
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

### 5.7 `cost/total_cost_usd`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 总成本（美元）

- **计算**: `input_cost_usd + output_cost_usd`
- **用途**: 追踪实验总成本
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图（单调递增）

**示例**:
```python
# 100K input tokens + 50K output tokens with GPT-4
cost/input_tokens: 100000
cost/output_tokens: 50000
cost/input_cost_usd: 3.00    # 100K * $30/1M
cost/output_cost_usd: 3.00   # 50K * $60/1M
cost/total_cost_usd: 6.00
```

### 5.8 `cost/avg_tokens_per_call`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 平均每次 API 调用的 Token 数

- **计算**: `total_tokens / total_calls`
- **用途**: 了解平均调用规模
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图

### 5.9 `cost/avg_cost_per_call`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 平均每次 API 调用的成本（美元）

- **计算**: `total_cost_usd / total_calls`
- **用途**: 了解单次调用成本
- **更新时机**: 每次 LLM 调用后
- **图表类型**: 折线图

---

## 6. Islands (岛屿指标)

多岛屿实验的每个岛屿的独立指标（仅在 `num_islands > 1` 时记录）。

### 全局岛屿指标

### 6.1 `islands/num_islands`

**类型**: 整数
**范围**: [1, ∞)
**说明**: 岛屿总数

- **用途**: 了解实验规模
- **更新时机**: 每次迭代后
- **图表类型**: 常数（水平线）

### 6.2 `islands/total_population`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 所有岛屿的种群总数

- **计算**: `sum(island_0_size, island_1_size, ...)`
- **用途**: 追踪总种群规模
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

### 6.3 `islands/global_best_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 所有岛屿中的最佳分数

- **计算**: `max(island_0_best, island_1_best, ...)`
- **用途**: 追踪全局最佳性能
- **更新时机**: 每次迭代后
- **图表类型**: 折线图（单调不递减）

### 6.4 `islands/best_island`

**类型**: 整数
**范围**: [0, num_islands-1]
**说明**: 当前最佳分数所在的岛屿 ID

- **用途**: 追踪哪个岛屿表现最好
- **更新时机**: 每次迭代后
- **图表类型**: 离散点图

### 每个岛屿的指标

对于每个岛屿 `island_id`（0, 1, 2, ...），记录以下指标：

#### 6.5 `islands/island_{id}/size`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的种群大小

- **用途**: 对比不同岛屿的规模
- **更新时机**: 每次迭代后
- **图表类型**: 折线图（每个岛屿一条线）

**示例**:
```
islands/island_0/size: 50
islands/island_1/size: 45
islands/island_2/size: 48
```

#### 6.6 `islands/island_{id}/best_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的最佳分数

- **用途**: 对比不同岛屿的性能
- **更新时机**: 每次迭代后
- **图表类型**: 折线图（每个岛屿一条线）

**WandB 中的趋势图**:
- X轴: iteration
- Y轴: best_score
- 每个岛屿一条线
- 可以清楚看到哪个岛屿表现最好

**示例**:
```
islands/island_0/best_score: 2.5  ← Island 0 领先
islands/island_1/best_score: 2.3  ← Island 1 紧随其后
islands/island_2/best_score: 1.8  ← Island 2 稍落后
```

#### 6.7 `islands/island_{id}/avg_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的平均分数

- **计算**: `mean(island_{id}_scores)`
- **用途**: 对比不同岛屿的整体水平
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

#### 6.8 `islands/island_{id}/median_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的中位数分数

- **用途**: 了解岛屿分数的中间值
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

#### 6.9 `islands/island_{id}/std_score`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的分数标准差

- **用途**: 衡量岛屿内的分数离散程度
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

#### 6.10 `islands/island_{id}/best_generation`

**类型**: 整数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 中最佳程序的代数

- **用途**: 追踪岛屿的最佳演化代数
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

#### 6.11 `islands/island_{id}/avg_generation`

**类型**: 浮点数
**范围**: [0, ∞)
**说明**: 岛屿 {id} 的平均代数

- **计算**: `mean(island_{id}_generations)`
- **用途**: 追踪岛屿的演化深度
- **更新时机**: 每次迭代后
- **图表类型**: 折线图

#### 6.12 `islands/island_{id}/best_program_id`

**类型**: 字符串
**说明**: 岛屿 {id} 最佳程序的 ID

- **用途**: 识别最佳程序
- **更新时机**: 每次迭代后
- **图表类型**: 文本表格

---

## 指标使用示例

### Circle Packing 实验

在 Circle Packing 任务中，指标会是这样演化的：

```
Iteration 0 (初始化):
  iteration/iteration: 0
  performance/best_score: 0.50
  performance/avg_score: 0.45
  population/size: 10
  diversity/genetic_diversity: 0.0

Iteration 5:
  iteration/iteration: 5
  performance/best_score: 1.20  ← 改进
  performance/avg_score: 0.95  ← 提升
  population/size: 50
  diversity/genetic_diversity: 0.6

Iteration 10:
  iteration/iteration: 10
  performance/best_score: 2.50  ← 接近最优
  performance/avg_score: 2.10  ← 整体优秀
  population/size: 100
  diversity/genetic_diversity: 0.8
```

---

## WandB 仪表板布局

在 WandB 仪表板中，指标会自动分组：

### Charts 标签页

**Performance**:
- performance/best_score
- performance/avg_score
- performance/median_score
- performance/std_score
- performance/p10_score
- performance/p90_score

**Population**:
- population/size
- population/avg_generation
- population/max_generation
- population/min_generation
- population/unique_programs
- population/unique_ids

**Diversity**:
- diversity/genetic_diversity
- diversity/code_diversity
- diversity/parent_diversity

**Iteration**:
- iteration/iteration

**Cost** (如果启用):
- cost/total_calls
- cost/total_tokens
- cost/total_cost_usd
- ...

---

## 自定义指标

如果需要添加自定义指标，可以在代码中添加：

```python
# 在 Evolver._record_metrics() 中
metrics["custom/my_metric"] = calculate_custom_metric()

# 或者在 LLM 客户端中记录成本
cost_tracker.record_call(input_tokens=1000, output_tokens=500)
```

---

## 指标采样频率

- **默认**: 每次迭代后记录
- **初始状态**: Iteration 0（初始程序评估后）也会记录
- **配置**: 通过 `checkpoint_interval` 调整（如果实现）
- **异步**: 非阻塞记录，不影响性能

**记录时机**:
1. **Iteration 0**: 初始程序加载并经过 enrichment（评估）后记录
2. **Iteration 1-N**: 每次 rollout 完成后记录
3. **实验结束**: 最终状态记录

---

## 数据保留

- WandB 自动保存所有历史数据
- 可以随时查看过去的实验
- 支持实验对比和分组

---

## 性能影响

- **异步记录**: < 1% 性能影响
- **网络带宽**: 每次约 1-5 KB
- **存储**: WandB 云端存储

---

## 故障排除

### Q: 某些指标没有显示？

**A**: 检查以下几点：
1. 程序是否有有效的 `combined_score`？
2. 种群是否为空？
3. 查看日志中是否有 "Failed to record metrics" 警告

### Q: Cost 指标都是 0？

**A**: Cost 追踪需要在 LLM 客户端中集成 `CostTracker`，目前可能未实现。

### Q: 指标更新延迟？

**A**: 使用异步模式，指标可能在几秒钟后才出现在 WandB。

---

## 总结

监控系统自动记录 **18 个核心指标**：

- 1 个迭代指标
- 7 个性能指标
- 6 个种群指标
- 4 个多样性指标（包括语义多样性）✨
- 若干成本指标（如果启用）

这些指标提供了进化过程的完整视图，帮助你：
- 🎯 追踪性能改进
- 📊 分析种群结构
- 🎨 评估多样性（结构、语法、语义三个层面）
- 💰 监控成本
- 📈 对比不同实验

**多样性指标的三个层面**:
1. **Genetic Diversity** (基因多样性) - 结构层面：不同世代的比例
2. **Code Diversity** (代码多样性) - 语法层面：代码文本的重复程度
3. **Parent Diversity** (父代多样性) - 谱系层面：不同父代的比例
4. **Semantic Diversity** (语义多样性) - 行为层面：程序功能的差异 ⭐

---

**文档版本**: 1.0
**最后更新**: 2026-01-29
**相关文件**:
- `famou/infrastructure/monitor/metrics_collector.py`
- `famou/controller/evolver.py`
- `famou/infrastructure/monitor/wandb_monitor.py`
