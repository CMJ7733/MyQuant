# 监控系统使用指南

## 快速开始

### 1. 在配置文件中启用监控

在你的 `config.yaml` 中添加以下配置：

```yaml
infrastructure:
  # ... 其他配置 ...

  # 监控配置
  monitor:
    enabled: true  # 启用监控
    type: wandb  # 监控后端类型

    wandb:
      project: my-project  # WandB 项目名称
      tags:
        - experiment-tag
      notes: "实验描述"

      # 异步日志配置（推荐）
      async_mode: true  # 非阻塞日志，不影响进化速度
      num_workers: 2
      queue_size: 100

      # 监控内容
      track_metrics:
        performance: true  # 性能指标（最佳分数、平均分数等）
        population: true  # 种群指标（大小、代数等）
        diversity: true  # 多样性指标
        cost: true  # LLM API 成本
      track_artifacts: true  # 记录 artifact（最佳程序等）
      track_lineage_tree: true  # 记录血缘树
      checkpoint_interval: 1  # 每 N 次迭代记录一次
```

### 2. 安装 WandB

```bash
pip install wandb
```

### 3. 认证 WandB

```bash
wandb login
```

### 4. 运行实验

```bash
python -m famou \
    --config examples/circle_packing/config.yaml \
    --programs examples/circle_packing/init.py \
    --evaluator examples/circle_packing/evaluator.py
```

监控会自动启用，指标会实时上传到 WandB。

## 配置选项详解

### monitor.enabled
- **类型**: `bool`
- **默认**: `false`
- **说明**: 是否启用监控。设置为 `false` 可以完全禁用监控系统。

### monitor.type
- **类型**: `str`
- **默认**: `"wandb"`
- **说明**: 监控后端类型。目前只支持 `"wandb"`。

### monitor.wandb.project
- **类型**: `str`
- **默认**: `"famou-experiments"`
- **说明**: WandB 项目名称。所有实验会记录到这个项目下。

### monitor.wandb.entity
- **类型**: `str` 或 `null`
- **默认**: `null`
- **说明**: WandB 实体（用户名或团队名称）。`null` 表示使用你的默认账户。

### monitor.wandb.tags
- **类型**: `list[str]`
- **默认**: `[]`
- **说明**: 实验标签，用于组织和筛选实验。

### monitor.wandb.notes
- **类型**: `str` 或 `null`
- **默认**: `null`
- **说明**: 实验备注说明。

### monitor.wandb.async_mode
- **类型**: `bool`
- **默认**: `true`
- **说明**: 是否启用异步日志。
  - `true`: 非阻塞，不影响进化速度（推荐）
  - `false`: 同步日志，每次记录都会等待

### monitor.wandb.num_workers
- **类型**: `int`
- **默认**: `2`
- **说明**: 异步日志的工作线程数。

### monitor.wandb.queue_size
- **类型**: `int`
- **默认**: `100`
- **说明**: 异步日志队列的最大大小。

### monitor.wandb.track_metrics
- **类型**: `dict`
- **默认**:
  ```yaml
  performance: true
  population: true
  diversity: true
  cost: true
  ```
- **说明**: 控制要记录的指标类型。

### monitor.wandb.track_artifacts
- **类型**: `bool`
- **默认**: `true`
- **说明**: 是否记录 artifacts（如最佳程序代码）。

### monitor.wandb.track_lineage_tree
- **类型**: `bool`
- **默认**: `true`
- **说明**: 是否记录血缘树可视化。

### monitor.wandb.checkpoint_interval
- **类型**: `int`
- **默认**: `1`
- **说明**: 每 N 次迭代记录一次指标。

## 查看监控结果

### 在 WandB 仪表板中查看

1. 打开浏览器访问: https://wandb.ai/
2. 找到你的项目（如 `famou-circle-packing`）
3. 点击运行名称查看详细图表

### 可视化内容

#### 1. 性能图表
- `performance/best_score`: 最佳分数
- `performance/avg_score`: 平均分数
- `performance/median_score`: 中位数分数
- `performance/std_score`: 标准差

#### 2. 种群图表
- `population/size`: 种群大小
- `population/avg_generation`: 平均代数
- `population/unique_programs`: 唯一程序数

#### 3. 多样性图表
- `diversity/genetic_diversity`: 基因多样性
- `diversity/code_diversity`: 代码多样性

#### 4. 成本图表
- `cost/total_calls`: LLM 调用次数
- `cost/total_tokens`: 总 Token 数
- `cost/total_cost_usd`: 总成本（美元）

#### 5. 血缘树可视化
- 交互式树状图展示程序演化历史

## 配置示例

### 示例 1: 最小配置

```yaml
infrastructure:
  monitor:
    enabled: true
    wandb:
      project: my-experiments
```

### 示例 2: 生产环境配置

```yaml
infrastructure:
  monitor:
    enabled: true
    wandb:
      project: production-experiments
      entity: my-team
      tags:
        - production
        - circle-packing
      notes: "生产环境实验"
      async_mode: true
      num_workers: 4
      queue_size: 200
```

### 示例 3: 调试配置（同步模式）

```yaml
infrastructure:
  monitor:
    enabled: true
    wandb:
      project: debug-experiments
      async_mode: false  # 同步模式，更容易调试
```

### 示例 4: 禁用监控

```yaml
infrastructure:
  monitor:
    enabled: false  # 完全禁用监控
```

## 常见问题

### Q: 如何禁用监控？
A: 在配置文件中设置 `monitor.enabled: false`

### Q: 监控会影响性能吗？
A: 如果使用 `async_mode: true`（默认），对性能影响极小。日志操作在后台线程异步执行。

### Q: 如何查看实时成本？
A: 在 WandB 仪表板中查看 `cost/total_cost_usd` 指标。

### Q: 可以不安装 wandb 吗？
A: 可以。如果不安装 wandb，监控会自动禁用，不影响实验运行。

### Q: 如何组织多次实验？
A: 使用 `tags` 和 `group` 字段：
```yaml
wandb:
  project: my-experiments
  group: baseline-comparison  # 相同 group 的实验会在一起
  tags:
    - strategy-a
    - test-1
```

## 高级用法

### 自定义指标收集

你可以在代码中直接使用监控系统：

```python
from famou.infrastructure.monitor import EvolutionMetricsCollector

# 创建收集器
collector = EvolutionMetricsCollector()

# 收集指标
metrics = collector.collect_from_experiment(experiment)

# 添加自定义指标
metrics["custom/my_metric"] = calculate_custom_metric()

# 记录到监控
monitor.log_metrics(metrics, step=iteration)
```

### 成本追踪

```python
from famou.infrastructure.monitor import CostTracker

# 创建成本追踪器
cost_tracker = CostTracker(model_name="gpt-4")

# 记录每次 LLM 调用
cost_tracker.record_call(
    input_tokens=llm_response.usage.prompt_tokens,
    output_tokens=llm_response.usage.completion_tokens
)

# 获取成本
cost_metrics = cost_tracker.get_metrics()
```

## 更多信息

- [WandB 文档](https://docs.wandb.ai/)
- [使用示例](EXAMPLES.md)
- [实现计划](IMPLEMENTATION_PLAN.md)
