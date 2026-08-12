# Famou 2.0 - LLM驱动的演化代码优化框架

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-green.svg)](https://docs.pydantic.dev/)

Famou 2.0 是基于 [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 理念构建的新一代 LLM 代码演化框架。通过迭代式的 Rollout 流程（选择 → 生成 → 评估 → 反馈），实现程序的自动化演化优化。

## 核心特性

### 模块化流水线架构
- **Select**: 从种群中选择父代程序（支持 Elite、Cluster、Random 等策略）
- **Generate**: 使用 LLM 对父代进行变异生成子代（支持 Mutation、Crossover）
- **Evaluate**: 运行用户自定义评估器计算适应度
- **Judge**: 丰富程序反馈信息，策略相关多轮信息
- **自定义模块**: 支持扩展任意反馈和处理模块

### 多岛屿演化模型
- **独立演化**: 多个岛屿并行演化，保持种群多样性
- **迁移机制**: 支持 Ring 拓扑的岛屿间程序迁移
- **重置策略**: 定期重置表现差的岛屿，防止局部最优

### 高性能并发

Famou 2.0 支持两种并发执行后端：`ThreadPoolBackend` 和 `RayBackend`，可通过配置灵活切换。

#### ThreadPoolBackend（默认）

使用 Python `ThreadPoolExecutor` 实现多线程并发执行，适合单机环境：

```yaml
infrastructure:
  backend:
    mode: threadpool
```

**特点**：
- 无需额外依赖，开箱即用
- 每个 Worker 有独立的线程本地环境
- 使用独立的 LLM 客户端池，避免连接池竞争

#### RayBackend

使用 Ray 分布式计算框架，支持集群环境下的分布式执行和故障恢复。

**依赖安装**：
```bash
pip install ray==2.46.0
```

**两种连接模式**：

##### 模式 1: Ray Client Proxy（直连集群）

在配置文件中指定 `ray_host` 为 `ray://` 地址，框架会自动上传源码和依赖到远端集群：

```yaml
infrastructure:
  backend:
    mode: ray
    ray_host: ray://<head_node_ip>:<head_node_port>
```

##### 模式 2: Ray Job 提交（推荐）

通过 `run_famou_ray_job.py` 脚本将实验作为 Ray Job 提交到集群。脚本会自动完成源码上传、依赖安装、日志实时输出和信号处理。

配置文件中 `ray_host` 设为 `auto`：

```yaml
infrastructure:
  backend:
    mode: ray
    ray_host: auto  # Ray Job 模式，由提交脚本管理集群连接
```

**通用提交脚本**：
```bash
# 直接指定参数
python run_famou_ray_job.py \
  -c examples/circle_packing/config_ray.yaml \
  -p examples/circle_packing/init.py \
  -e examples/circle_packing/evaluator.py \
  --ray-address http://<ray_dashboard_ip>:8265 \
  --job-name my-experiment

# 或通过环境变量
RAY_ADDRESS=http://<ray_dashboard_ip>:8265 \
JOB_NAME=my-experiment \
python run_famou_ray_job.py -c config.yaml -p init.py -e evaluator.py
```

**示例快捷脚本**
```bash
# Circle Packing 示例
bash examples/circle_packing/run_ray_job.sh

# 覆盖集群地址或任务名
RAY_ADDRESS=http://other:8265 JOB_NAME=test-v1 \
  bash examples/circle_packing/run_ray_job.sh
```

**Ray Job 模式特点**：
- 自动从 `requirements.txt` 提取 pip 依赖和源配置，构建 `runtime_env`
- 通过 `py_modules` 上传 `famou/` 源码到集群
- 提交后实时流式输出远端日志，任务结束后自动退出
- 收到 Ctrl+C / SIGTERM 时自动取消远端 Ray Job
- 支持自定义任务名，可通过 `ray job logs/status/stop <job_name>` 管理

**任务管理**：
```bash
ray job status <job_name>   # 查看任务状态
ray job logs <job_name>     # 查看日志
ray job stop <job_name>     # 停止任务
```

**特点**：
- 支持分布式集群执行
- 原生支持 GPU 资源分配
- 内置 Actor 故障恢复机制（默认 max_restarts=3）
- 可配置 CPU/GPU 资源：

```yaml
infrastructure:
  env:
    type: local
    cpu: 1
    gpu: 1  # GPU数量
```

#### 并发配置

在实验配置中控制并发度：

```yaml
experiment:
  max_workers: 4  # 并发Worker数量
```

### 实验管理
- **实验配置**: YAML 格式配置文件，支持多初始解并行演化
- **断点续传**: 智能检查点机制，支持实验恢复和自动分叉
- **Compact 存储**: 10-100x 压缩率的紧凑存储格式（Archive + ID 引用）
- **自动 Fork**: 配置变更或指定迭代恢复时自动创建新实验分支

### 灵活的基础设施
- **LLM 客户端**: 支持 OpenAI 兼容接口; TODO: 更多接口风格支持，Router支持
- **数据存储**: 本地JSON 格式存储程序、结果和检查点；TODO: 支持远程数据库存储
- **结构化日志**: 本地同时输出控制台、文本日志和 JSONL 格式; TODO: 支持远程多机日志服务

## 系统架构

```
run_famou.py
  -> Evolver (实验生命周期管理、流式执行)
     -> RolloutEngine (流水线执行、依赖注入、重试)
        -> Rollout (Select -> Generate -> Evaluate -> Judge -> etc)
     -> PopulationModule (将新程序合并到种群)
     -> DataService (持久化程序/结果/检查点)
```

核心数据对象：
```
Experiment (全局实验状态)
  -> Context (单次 Rollout 的只读快照)
  -> RolloutResult (单次 Rollout 的可变工作状态)
  -> Program (单个候选解)
```

## 快速开始

### 环境配置

#### 1. 安装依赖
```bash
# 推荐使用 Python 3.12 及conda，uv等环境管理工具
pip install -r requirements.txt
```

#### 2. 配置 LLM API
在配置文件中设置你的 LLM API 密钥：
```yaml
infrastructure:
  llm:
    provider: openai
    api_base: https://your-api-endpoint
    model: your-model-name
    api_key: your-api-key
```

### 运行方式

安装后有三种运行方式：

```bash
# 方式 1: 命令行工具
famou-run -c config.yaml -p init_1.py init_2.py -e evaluator.py

# 方式 2: 模块方式
python -m famou -c config.yaml -p init_1.py init_2.py -e evaluator.py

# 方式 3: 脚本方式（推荐，无需安装）
python run_famou.py -c config.yaml -p init_1.py init_2.py -e evaluator.py
```

### 运行示例

#### Circle Packing 示例
```bash
python run_famou.py \
  -c examples/circle_packing/config.yaml \
  -p examples/circle_packing/init.py \
  -e examples/circle_packing/evaluator.py
```

#### 恢复实验
```bash
# 从最新检查点恢复（继续同一实验）
python run_famou.py \
  --resume famou_data/circle_packing_20260119_143022_a1b2 \
  -e examples/circle_packing/evaluator.py

# 从指定迭代恢复（自动分叉为新实验）
python run_famou.py \
  --resume famou_data/circle_packing_20260119_143022_a1b2 \
  --iteration 5 \
  -e examples/circle_packing/evaluator.py

# 使用新配置恢复（自动分叉为新实验）
python run_famou.py \
  --resume famou_data/circle_packing_20260119_143022_a1b2 \
  -c new_config.yaml \
  -e examples/circle_packing/evaluator.py
```

## 创建自定义任务

可以参考 https://ku.baidu-int.com/d/Y2AUu3qHCW3_9m 进行初始解/评估器/提示词的撰写

### 1. 编写初始程序

创建包含 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 标记的初始程序：

```python
# EVOLVE-BLOCK-START
def my_algorithm():
    """
    你的算法实现
    LLM 会演化这个代码块
    """
    # 初始实现...
    return result
# EVOLVE-BLOCK-END

# 这部分不会被演化
if __name__ == "__main__":
    print(my_algorithm())
```

### 2. 编写评估器

评估器需实现 `evaluate(path_user_py)` 函数：

```python
def evaluate(path_user_py):
    """
    评估程序的适应度

    Args:
        path_user_py: 程序文件路径

    Returns:
        dict: 必须包含以下字段
            - combined_score (float): 综合评分，越高越好
            - validity (float): 有效性，0.0 或 1.0
            - 其他自定义指标...
    """
    # 加载并运行程序
    # 计算评分
    return {
        "combined_score": score,
        "validity": 1.0 if valid else 0.0,
        "custom_metric": value,
    }
```

### 3. 创建配置文件

```yaml
# config.yaml
experiment:
  name: my_experiment
  task_description: |
    # 任务描述
    详细描述你的优化目标...

  language: python
  strategy: standard
  max_iterations: 100
  max_workers: 4
  checkpoint_interval: 10
  seed: 42

  island:
    num_islands: 3
    island_size: 20
    migration_interval: 10
    migration_size: 2
    migration_topology: ring

infrastructure:
  llm:
    provider: openai
    api_base: https://your-api
    model: your-model
    api_key: your-key
    temperature: 0.7
    max_tokens: 16000

  storage:
    type: local
    base_path: ./famou_data

  logger:
    type: local
    level: INFO
```

### 4. 运行演化

```bash
python run_famou.py \
  -c config.yaml \
  -p init.py \
  -e evaluator.py
```

## 目录结构

```
famou_v2/
├── run_famou.py                 # 入口脚本
├── run_famou_ray_job.py         # Ray Job 提交脚本（提交到远程集群）
├── famou/
│   ├── core/
│   │   ├── data.py              # 核心数据模型 (Program, Context, Experiment)
│   │   ├── protocol.py          # 协议定义
│   │   └── types.py             # 类型定义
│   ├── controller/
│   │   ├── engine.py            # RolloutEngine (流水线执行)
│   │   └── evolver.py           # Evolver (实验生命周期)
│   ├── modules/
│   │   ├── select/              # 选择模块 (Elite, Cluster, Random)
│   │   ├── generate/            # 生成模块 (Mutation, Crossover)
│   │   ├── evaluate/            # 评估模块
│   │   ├── judge/               # 判断模块 (LLM Judge)
│   │   └── population/          # 种群管理 (TopK, Cluster)
│   ├── strategies/
│   │   ├── _registry.py         # 策略注册表（自动发现）
│   │   ├── ...                  # TopK+聚类策略
│   │   └── strategy.py # 简单经验策略
│   ├── infrastructure/
│   │   ├── llm/                 # LLM 客户端
│   │   ├── env/                 # 执行环境
│   │   ├── storage/             # 存储服务
│   │   ├── logger/              # 日志服务
│   │   └── embedding/           # 嵌入服务
│   ├── prompts/
│   │   └── templates/           # Jinja2 提示词模板
│   └── config/
│       └── settings.py          # 配置模型
├── examples/
│   └── circle_packing/          # Circle Packing 示例
└── tests/                       # 测试用例
```

## 数据存储结构

```
famou_data/
└── {experiment_id}/
    ├── config.yaml                                     # 实验配置
    ├── experiment.log                                  # 文本日志
    ├── experiment.jsonl                                # 结构化日志
    ├── experiment_checkpoint_{iter}.json               # 检查点（Compact 格式）
    ├── programs/
    │   ├── {iter}_{gen}_{hash}.json                    # 程序元数据（完整对象）
    │   └── {iter}_{gen}_{hash}.py                      # 程序代码
    └── results/
        └── {exp_id}_island_{island}_rollout_{iter}.json # Rollout 结果（Compact 格式）
```

**Compact 存储设计**：
- **Archive** 存储完整 Program 对象（单一数据源）
- **Populations** 和 **RolloutResults** 仅存储 Program ID
- 加载时通过 ID 在 Archive 中查找重建
- 存储空间减少 10-100 倍（50 程序 × 3 岛屿：145MB → 1.5MB）

## 扩展指南

### 添加新模块
在 `famou/modules/` 下创建新模块，实现 `execute(context, result)` 方法：

```python
from famou.modules import Module

class MyModule(Module):
    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        # 实现逻辑
        return result
```

### 添加新策略
在 `famou/strategies/` 下创建策略文件（如 `my_strategy.py`）：

```python
def create_strategy(evaluate_fn, params):
    from famou.controller.engine import Rollout
    from famou.modules import TopKSelect, MutationGenerate, EvaluateModule
    from famou.modules.population import TopKPopulation

    rollout = Rollout(
        modules=[
            TopKSelect(name="select"),
            MutationGenerate(name="generate"),
            EvaluateModule(name="evaluate", evaluate_fn=evaluate_fn),
        ]
    )
    # 构建种群管理器
    population = TopKPopulation()

    return {
        "rollout": rollout,
        "population_module": population,
        "description": "自定义策略描述",
        "tags": ["exploration", "custom"],
    }
```

策略会被自动发现并注册，在配置文件中使用：
```yaml
experiment:
  strategy: my_strategy  # 使用文件名（不带 .py）
```

### 添加新提示词模板
在 `famou/prompts/templates/` 下添加 Jinja2 模板文件。

## 维护团队

### 核心维护者
- **wuchufan** (wuchufan@baidu.com)
- **gezengle** (gezengle@baidu.com)
- **xiajing** (xiajing05@baidu.com)
- **zhaomo** (zhaomo01@baidu.com)
- **fuyin** (fuyin@baidu.com)


### 项目所有者
- **liannan** (liannan@baidu.com)