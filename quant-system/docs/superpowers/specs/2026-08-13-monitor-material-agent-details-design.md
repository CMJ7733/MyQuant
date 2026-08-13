# CogAlpha Monitor Material 3 与 Agent 详情设计

日期：2026-08-13
状态：已获用户设计批准，待书面规格复核

## 1. 目标

重新设计 CogAlpha monitor，使它：

1. 使用接近 Google Material 3 / Google Cloud 的浅色视觉语言，提升长时间监控时的可读性与美观度。
2. 允许用户点击全部 21 个 agent，查看该 agent 的职责、状态、指标、每代结果和实际操作时间线。
3. 保留现有轻量、离线友好的单 HTML + FastAPI + SSE 架构，不引入 npm、前端框架、构建步骤或外部 CDN。

## 2. 已确认的产品决定

- 视觉方向：Material 3 浅色蓝（Cloud Blue）。
- Agent 详情交互：从右侧打开约 60% 宽的详情面板。
- 信息深度：运行概览 + 实时活动时间线 + 按需展开深层详情。
- 界面语言：中文为主；Agent 名称和技术字段保留英文。
- 覆盖范围：全部 21 个 agent，包括本次运行未选中或尚未运行的 agent。
- 数据方案：新增 Agent 专用详情接口，不把完整历史或 prompt/response 塞入 SSE 快照。

## 3. 范围

### 3.1 包含

- 重做 monitor 的配色、排版、卡片、表格、状态标签、图表和响应式布局。
- 调整全局总览的信息层级和中文文案。
- 新增按 agent 聚合详情的后端读取能力与 HTTP API。
- 新增 Agent 详情面板的概览、活动和每代结果三个标签页。
- 复用现有 call、generation、alpha 和 candidate 深层详情能力。
- 增加必要的 reader/API 测试和前端渲染验证。

### 3.2 不包含

- 不改变搜索、进化、质量检查或归档写入流程。
- 不改变 agent 的选择算法、层级定义或执行顺序。
- 不引入用户账户、权限系统、数据库或远端服务。
- 不提供运行控制，例如暂停、停止、重试或修改配置。
- 不引入 React、Vue、npm、CDN 字体或第三方图表库。
- 不将 prompt/response 预加载进 Agent 详情或 SSE。

## 4. 现有系统约束

现有 monitor 由以下部分组成：

- `cogalpha/monitor/reader.py` 增量读取 `generations.jsonl` 与 `llm_calls.jsonl`，并聚合全局快照。
- `cogalpha/monitor/server.py` 提供静态页面、SSE 快照和按需详情接口。
- `cogalpha/monitor/static/index.html` 包含所有 HTML、CSS、原生 JavaScript 和手写 SVG 图表。

`llm_calls.jsonl` 可能达到数 GB，因此 reader 在聚合时必须丢弃 prompt/response 正文；正文只允许在用户点击单次调用后读取。

## 5. 架构与组件边界

### 5.1 RunReader

`RunReader` 继续负责从归档构建可序列化的读取模型，不依赖搜索运行时对象。

新增一个面向 agent 的公开方法，概念签名如下：

```python
def agent_detail(self, name: str) -> Optional[Dict[str, Any]]
```

职责：

- 验证 `name` 是否属于已知 21-agent 层级。
- 返回该 agent 的静态身份、选择状态与当前运行状态。
- 聚合该 agent 的 generation 记录和 LLM call 摘要。
- 生成详情页所需的指标、趋势与时间线，不包含 prompt/response 正文。

reader 在增量处理 call 时维护以下有界状态：

- 每个 agent 最近 200 条轻量 call 摘要的 `deque`。
- 每个 agent 的 call、Token 和 latency 累计计数。

这些状态在现有 `_apply_call` 路径中以 O(1) 代价更新。`agent_detail` 从内存中的 generation 记录、有界 call 摘要和累计计数组装响应，不重新扫描大型 `llm_calls.jsonl`。

它不负责 HTML、文案或视觉状态。

### 5.2 HTTP API

新增：

```text
GET /api/agent/{name}
```

行为：

- 已知 agent：返回详情对象；未运行时仍返回 200 和空数据集合。
- 未知 agent：返回 404。
- 返回内容不含完整 prompt/response。

现有接口继续使用：

- `GET /api/call/{seq}`：加载单次完整 prompt/response。
- `GET /api/generation/{agent}/{generation}`：加载某代 alpha 列表。
- `GET /api/alpha/{id}`：加载 alpha 代码、检查、适应度和 lineage。
- `GET /api/candidate/{file}`：加载归档候选代码。

### 5.3 前端 Dashboard

前端仍为单文件原生 JavaScript。建议保持以下逻辑边界：

- 全局快照渲染：顶部状态、指标卡、Agent 层级、漏斗、趋势、成本和候选结果。
- Agent 面板控制：打开、关闭、当前标签、加载与刷新。
- Agent 详情渲染：概览、活动、每代结果。
- 深层详情：复用现有 call/generation/alpha/candidate drawer 内容，但在同一详情表面中进行层级切换或返回，避免叠加多个不可管理的抽屉。

静态本地化由前端常量负责：七层名称、21 个 agent 的简短中文职责和界面文案使用中文；API 同时返回 `AgentSpec.layer`、`focus` 与 `probe` 的原始英文内容，供详情中的“研究定义”折叠区展示。运行数据和领域定义不在 reader 中翻译。

## 6. 数据模型

Agent 详情响应至少包含：

```json
{
  "name": "AgentMarketCycle",
  "display_name": "MarketCycle",
  "level": 1,
  "level_name": "Market cycle",
  "description": "识别长期趋势、市场阶段和周期状态转换。",
  "selected": true,
  "status": "running",
  "current_generation": 12,
  "current_cycle": 3,
  "summary": {
    "generations": 12,
    "generated": 146,
    "passed": 84,
    "qualified": 28,
    "elite": 6,
    "best_score": 0.086,
    "best_rank_ic": 0.071,
    "llm_calls": 186,
    "llm_tokens": 326000,
    "seconds": 742.3,
    "stopped_early": null
  },
  "trajectory": [],
  "generations": [],
  "recent_operations": []
}
```

### 6.1 Generation 摘要

每代记录包含：

- generation、cycle 和时间/耗时信息（归档有数据时）。
- n_generated、n_passed、n_qualified、n_elite。
- elite_mean_score 和 best 摘要。
- reject_counts 与 op_counts。
- llm_calls 与 wall_seconds。

### 6.2 Operation 时间线

每条操作包含：

- seq、role、generation、cycle、mode。
- tokens、latency_ms、response 字符数和可用的时间信息。
- 可用于加载 `/api/call/{seq}` 的稳定标识。

只返回该 agent 最近 200 条调用摘要，不提供任意历史分页。这个边界符合已选择的“实时活动时间线”范围，并防止大型运行的内存和传输成本随调用总量无限增长。完整历史仍保留在原始 JSONL 归档中，但不属于本次交互设计。

## 7. 数据流与实时更新

1. 页面加载后继续连接 `/api/stream`，每秒接收全局轻量快照。
2. 用户点击任意 agent 卡片，前端打开骨架面板并请求 `/api/agent/{name}`。
3. 面板打开期间，全局 SSE 不停止。
4. 当 SSE 表明当前 agent 的 generation、调用数或状态变化时，前端对详情接口进行防抖刷新；不得每个 tick 无条件重扫大型 JSONL。
5. 用户点击一条 operation 时，再请求 `/api/call/{seq}`。
6. 用户点击某代时，请求现有 generation 接口；点击 alpha 时请求 alpha 接口。

详情刷新采用“变化触发 + 最短间隔”策略：只有所选 agent 的摘要字段变化时才刷新，且两次刷新间隔不小于 2 秒。

## 8. 视觉设计

### 8.1 色彩

- 页面背景：柔和蓝灰，接近 `#F8FAFD`。
- 卡片：白色。
- 主色：Google Blue，接近 `#0B57D0`。
- 主色浅背景：接近 `#D3E3FD`。
- 成功/完成：绿色，接近 `#137333` / `#E6F4EA`。
- 注意/警告：琥珀色；错误使用红色。
- 主文本：接近 `#202124`；次文本：接近 `#5F6368`。
- 边框：低对比蓝灰，接近 `#E1E5EB`。

正负金融指标继续显式显示正负号，并通过文字/符号辅助颜色，避免仅依赖色觉判断。配色从现有交易终端语义转为 Material 状态语义；具体正负色仍遵守项目当前的中国市场约定（红涨绿跌），但运行状态不复用这两个金融语义颜色。

### 8.2 形状与层级

- 主卡片圆角约 16px；小组件约 10–12px；状态标签为胶囊形。
- 使用细边框与轻阴影建立层级，不使用重阴影或高饱和大色块。
- 通过留白、字号和字重建立信息层级，减少全大写和等宽字体的面积。
- 动态数字保留 tabular numerals，防止实时更新时抖动。

### 8.3 页面结构

从上到下：

1. 应用栏：产品名、运行状态、运行名称、市场/模型摘要、最后更新时间。
2. 核心指标：Agent、Generation、LLM Calls、Tokens、Elite。
3. 七层 Agent 层级。
4. 质量漏斗与 Elite 趋势。
5. LLM 成本、近期活动与候选结果。

## 9. Agent 卡片

全部 21 个 agent 都可点击。

状态必须同时由文字和视觉标记表达：

- `running`：蓝色活动圆点、进度文案。
- `done`：绿色完成图标与“已完成”。
- `queued`：中性灰蓝与“排队中”。
- `unselected`：低强调度虚线边框与“本次未选中”，但不降低到不可读，也不禁用点击。

卡片展示 Agent 简称以及最有用的一行状态。只有存在真实运行数据时才显示 generation、qualified、elite 等数字。

## 10. Agent 详情面板

### 10.1 容器

- 桌面端从右侧打开，宽度约为视口的 60%，并设置合理最小/最大宽度。
- 保留背景 dashboard，使用户知道当前上下文。
- 支持关闭按钮、点击遮罩关闭和 `Escape` 关闭。
- 窄屏下占满视口。
- 打开后焦点进入面板；关闭后焦点返回原 Agent 卡片。

### 10.2 顶部

展示：

- Agent 名称、中文职责、层级。
- 选中状态和运行状态。
- 当前 generation/cycle（有数据时）。
- 关键计数的紧凑摘要。

### 10.3 标签页

#### 概览

- generated、passed、qualified、elite。
- LLM calls、Tokens、耗时。
- best score、best RankIC。
- Elite 轨迹。
- 当前操作摘要或明确的空状态。

#### 活动

- 按时间倒序显示该 agent 的操作时间线。
- 显示 role、generation/cycle、tokens、latency 和 mode。
- 最多展示最近 200 条记录，并明确标注这一范围。
- 点击记录加载完整 prompt/response。

#### 每代结果

- 每代的产出、通过、qualified、elite、耗时和漏斗摘要。
- 点击一代加载该代 alpha 列表。
- 点击 alpha 加载代码、检查结果、fitness、lineage 和淘汰原因。

## 11. 空状态与异常处理

- 未选中：解释该 agent 的职责，并标明“本次运行未选中，因此暂无运行数据”。
- 已选中但排队：显示“等待执行”，不显示误导性的零产出。
- 正在运行但暂无 generation：显示“已启动，等待首个 generation 记录”。
- 归档缺少 `alphas.jsonl`：说明运行中尚未保存完整 alpha 详情，保留 generation 摘要。
- 接口加载：使用骨架屏。
- 接口失败：面板内显示错误和重试按钮，不关闭面板，不影响全局 SSE。
- SSE 断开：沿用轮询回退，并明确显示连接状态。
- JSONL 损坏或存在半行：继续跳过坏记录或等待完整行，不让整个详情失败。

## 12. 安全与性能

- Agent 名称必须在已知层级白名单中，未知名称返回 404。
- 所有从归档读取并插入 HTML 的文本必须转义。
- prompt/response 只通过现有 call 详情接口按需读取。
- Agent 详情返回调用摘要，不返回正文。
- 每个 agent 的调用摘要使用固定 200 条的有界队列。
- 不在每次 SSE tick 无条件重扫归档。
- 保持默认绑定 `127.0.0.1` 的安全行为和现有网络暴露警告。

## 13. 测试与验收

### 13.1 Reader 与 API

- 已知 agent 能正确聚合其 generations 和 calls，不混入其他 agent。
- generation、call、token、耗时、qualified 和 elite 计数正确。
- 调用时间线倒序、200 条上限和有界队列淘汰行为正确。
- 未选中、排队、运行中和已完成四种状态返回正确。
- 未知 agent 返回 404。
- 缺少 `alphas.jsonl`、空 JSONL、半行和损坏行不导致接口崩溃。

### 13.2 前端

- 21 个 agent 均可点击。
- 打开、关闭、遮罩、Escape 和焦点返回行为正确。
- 三个标签页及其空状态正确渲染。
- API 失败显示重试状态。
- SSE 更新导致当前 agent 摘要变化时，详情按防抖策略刷新。
- 所有用户可见归档文本经过转义。

### 13.3 真实归档验收

使用项目现有完成归档和至少一个不完整归档启动 monitor，检查：

- 桌面宽屏和窄屏布局。
- 21 个 agent 的四类状态。
- 某个已有多代运行数据的 agent 的概览、最近活动和每代结果。
- 单次 call 的 prompt/response。
- generation → alpha → 代码/检查的深层导航。
- 无数据和缺少保存文件时的说明。
- SSE 或轮询更新期间面板不丢失选择和当前标签。

## 14. 成功标准

- 用户无需解析日志即可回答“哪个 agent 正在做什么、做了多少、效果如何、最近执行了哪些操作”。
- 全部 21 个 agent 都能打开详情，并且无数据状态不产生误导。
- 全局监控保持轻量，打开详情不会将完整调用历史加入 SSE。
- 界面在桌面和窄屏上保持可读，视觉一致符合已选择的 Material 3 浅色蓝方向。
- 现有 monitor 的 call、generation、alpha、candidate 深层查看能力继续可用。
