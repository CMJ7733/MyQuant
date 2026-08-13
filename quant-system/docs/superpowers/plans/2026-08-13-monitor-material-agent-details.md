# CogAlpha Monitor Material 3 与 Agent 详情 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 CogAlpha monitor 改造成 Material 3 浅色中文仪表盘，并让全部 21 个 agent 都能打开包含概览、最近活动和每代结果的实时详情面板。

**Architecture:** 保留 `RunReader → FastAPI/SSE → 单文件原生前端` 架构。`RunReader` 在现有增量 call 处理路径中，为每个 agent 维护有界 200 条摘要和累计计数；新增 `/api/agent/{name}` 按需返回详情，完整 prompt/response 继续通过现有 call 接口加载。前端重写视觉层，但保留现有全局数据与深层详情能力。

**Tech Stack:** Python 3.10+、pytest、FastAPI、httpx/TestClient、原生 HTML/CSS/JavaScript、SSE、手写 SVG。

**Design spec:** `docs/superpowers/specs/2026-08-13-monitor-material-agent-details-design.md`

**Repository note:** `/Users/edisonchen/Documents/quant/F4Q/quant-system` 当前不是 Git 仓库。以下每个 task 的测试检查点必须执行，但不创建 commit；如果执行前目录已被接入 Git，则可在每个 task 通过后按列出的文件范围提交。

---

## File map

- Create `tests/conftest.py`: monitor 归档夹具和 JSON/JSONL 写入辅助函数。
- Create `tests/test_monitor_agent_detail.py`: `RunReader.agent_detail` 的聚合、隔离、状态和有界队列测试。
- Create `tests/test_monitor_server.py`: Agent HTTP 接口的 200/404 行为。
- Create `tests/test_monitor_ui.py`: 静态页面的 Material tokens、中文结构、可访问性和详情入口保护测试。
- Modify `cogalpha/monitor/reader.py`: Agent 静态定义、per-agent call 累计、有界活动队列和详情响应。
- Modify `cogalpha/monitor/server.py`: 新增 Agent 详情路由。
- Modify `cogalpha/monitor/static/index.html`: Material 3 总览、21 个可点击 Agent、宽侧边详情面板和深层详情导航。
- Modify `pyproject.toml`: dev extra 增加 FastAPI TestClient 所需的 `httpx`。
- Modify `.gitignore`: 忽略视觉讨论产生的 `.superpowers/` 临时目录。
- Modify `README.md`: 更新 dashboard 说明和 Agent 详情行为。

---

### Task 1: 建立测试夹具并实现 Agent 详情聚合

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_monitor_agent_detail.py`
- Modify: `cogalpha/monitor/reader.py`

- [ ] **Step 1: 创建最小归档测试夹具**

写入 `tests/conftest.py`：

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def generation(
    agent: str = "AgentMarketCycle",
    generation_number: int = 0,
    cycle: int = 0,
    generated: int = 6,
    passed: int = 4,
    qualified: int = 2,
    elite: int = 1,
    calls: int = 3,
    seconds: float = 12.5,
) -> Dict[str, Any]:
    return {
        "generation": generation_number,
        "cycle": cycle,
        "agent": agent,
        "op_counts": {"hierarchy": generated},
        "n_generated": generated,
        "n_passed_checker": passed,
        "n_qualified": qualified,
        "n_elite": elite,
        "reject_counts": {"judge": generated - passed},
        "best": {"score": 0.086, "rank_ic": 0.071},
        "percentile_cutoffs": {},
        "elite_mean_score": 0.072,
        "llm_calls": calls,
        "wall_seconds": seconds,
    }


def call(
    seq: int,
    agent: str = "AgentMarketCycle",
    role: str = "generate",
    generation_number: int = 0,
    cycle: int = 0,
    tokens: int = 10,
) -> Dict[str, Any]:
    return {
        "seq": seq,
        "model": "mock-model",
        "temperature": 0.7,
        "system": "system",
        "prompt": f"prompt-{seq}",
        "response": f"response-{seq}",
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
        "finish_reason": "stop",
        "latency_ms": seq,
        "tags": {
            "role": role,
            "agent": agent,
            "generation": generation_number,
            "cycle": cycle,
            "mode": "light",
        },
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    write_json(
        tmp_path / "config.json",
        {
            "evolution": {
                "agents_per_run": 0,
                "seed": 42,
                "golden_ratio_selection": True,
                "generations": 24,
            }
        },
    )
    return tmp_path
```

- [ ] **Step 2: 写 Agent 身份、空状态和未知名称的失败测试**

写入 `tests/test_monitor_agent_detail.py`：

```python
from __future__ import annotations

from cogalpha.monitor.reader import RunReader

from conftest import call, generation, write_jsonl


def test_agent_detail_returns_static_identity_for_unselected_agent(run_dir):
    reader = RunReader(run_dir)
    reader.poll()

    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["name"] == "AgentMarketCycle"
    assert detail["display_name"] == "MarketCycle"
    assert detail["level"] == 1
    assert detail["layer"] == "Market Structure & Cycle Layer"
    assert "Long-term trends" in detail["focus"]
    assert detail["selected"] is False
    assert detail["status"] == "queued"
    assert detail["summary"]["generations"] == 0
    assert detail["generations"] == []
    assert detail["recent_operations"] == []


def test_agent_detail_rejects_unknown_agent(run_dir):
    reader = RunReader(run_dir)
    reader.poll()

    assert reader.agent_detail("AgentDoesNotExist") is None
```

- [ ] **Step 3: 运行测试并确认因缺少方法而失败**

Run:

```bash
python -m pytest tests/test_monitor_agent_detail.py -v
```

Expected: FAIL，错误包含 `AttributeError: 'RunReader' object has no attribute 'agent_detail'`。

- [ ] **Step 4: 写聚合与 Agent 隔离的失败测试**

追加到 `tests/test_monitor_agent_detail.py`：

```python
def test_agent_detail_aggregates_only_the_requested_agent(run_dir):
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(generation_number=0, generated=6, qualified=2, elite=1),
            generation(generation_number=1, generated=8, qualified=3, elite=2),
            generation(
                agent="AgentTailRisk",
                generation_number=0,
                generated=99,
                qualified=90,
                elite=80,
            ),
        ],
    )
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [
            call(1, tokens=10),
            call(2, role="judge", generation_number=1, tokens=20),
            call(3, agent="AgentTailRisk", tokens=999),
        ],
    )
    reader = RunReader(run_dir)
    reader.poll()

    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["selected"] is True
    assert detail["status"] == "done"
    assert detail["summary"] == {
        "generations": 2,
        "generated": 14,
        "passed": 8,
        "qualified": 5,
        "elite": 3,
        "best_score": 0.086,
        "best_rank_ic": 0.071,
        "llm_calls": 2,
        "llm_tokens": 30,
        "mean_latency_ms": 1.5,
        "seconds": 25.0,
        "stopped_early": None,
    }
    assert [row["generation"] for row in detail["generations"]] == [0, 1]
    assert [row["seq"] for row in detail["recent_operations"]] == [2, 1]
    assert all(row["agent"] == "AgentMarketCycle" for row in detail["recent_operations"])
    assert all("prompt" not in row and "response" not in row for row in detail["recent_operations"])


def test_agent_recent_operations_are_bounded_to_200(run_dir):
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq, tokens=1) for seq in range(1, 206)],
    )
    reader = RunReader(run_dir)
    reader.poll()

    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["summary"]["llm_calls"] == 205
    assert detail["summary"]["llm_tokens"] == 205
    assert len(detail["recent_operations"]) == 200
    assert detail["recent_operations"][0]["seq"] == 205
    assert detail["recent_operations"][-1]["seq"] == 6
```

说明：第三个 generation 切换到 `AgentTailRisk`，因此 `AgentMarketCycle` 状态应为 `done`。测试同时确保别的 agent 的 generation、Token 和 call 不会混入。

- [ ] **Step 5: 在 reader 中加入有界 per-agent call 状态**

在 `cogalpha/monitor/reader.py` 的常量区域加入：

```python
_RECENT_CALLS = 60
_AGENT_RECENT_CALLS = 200
```

在 `RunReader.__init__` 中、现有 `_recent` 初始化之后加入：

```python
self._agent_recent: Dict[str, Deque[Dict[str, Any]]] = {}
self._agent_call_counts: Counter = Counter()
self._agent_token_counts: Counter = Counter()
self._agent_latency_totals: Counter = Counter()
```

将 `_apply_call` 中活动摘要先保存为局部变量，再同时更新全局与 agent 状态：

```python
digest = {
    "seq": seq,
    "role": role,
    "agent": tags.get("agent"),
    "generation": tags.get("generation"),
    "cycle": tags.get("cycle"),
    "mode": tags.get("mode"),
    "temperature": record.get("temperature"),
    "model": record.get("model"),
    "tokens": tokens,
    "latency_ms": record.get("latency_ms"),
    "chars": len(record.get("response") or ""),
}
self._recent.append(digest)

agent_name = tags.get("agent")
if isinstance(agent_name, str) and agent_name:
    recent = self._agent_recent.setdefault(
        agent_name,
        deque(maxlen=_AGENT_RECENT_CALLS),
    )
    recent.append(digest)
    self._agent_call_counts[agent_name] += 1
    self._agent_token_counts[agent_name] += tokens
    self._agent_latency_totals[agent_name] += int(record.get("latency_ms", 0) or 0)
```

删除原来直接传给 `self._recent.append(...)` 的内联字典，避免同一摘要维护两份字段定义。

- [ ] **Step 6: 实现 Agent 详情响应**

在 `RunReader` 的 detail lookup 区域、`call_detail` 之前加入：

```python
def agent_detail(self, name: str) -> Optional[Dict[str, Any]]:
    """Return one known hierarchy agent's bounded monitoring detail."""
    from cogalpha.agents.hierarchy import HIERARCHY

    specs = {spec.name: spec for spec in HIERARCHY}
    spec = specs.get(name)
    if spec is None:
        return None

    agent = self._agents[name]
    records = [record for record in self._generations if record.get("agent") == name]
    last = records[-1] if records else {}
    call_count = int(self._agent_call_counts[name])
    mean_latency = (
        round(self._agent_latency_totals[name] / call_count, 1)
        if call_count
        else 0.0
    )

    return {
        "name": spec.name,
        "display_name": spec.name.replace("Agent", "", 1),
        "level": spec.level,
        "layer": spec.layer,
        "focus": spec.focus,
        "probe": spec.probe,
        "selected": agent.selected,
        "status": agent.status,
        "current_generation": last.get("generation"),
        "current_cycle": last.get("cycle"),
        "summary": {
            "generations": agent.generations,
            "generated": agent.n_generated,
            "passed": agent.n_passed,
            "qualified": agent.n_qualified,
            "elite": agent.n_elite,
            "best_score": agent.best_score,
            "best_rank_ic": agent.best_rank_ic,
            "llm_calls": call_count or agent.llm_calls,
            "llm_tokens": int(self._agent_token_counts[name]),
            "mean_latency_ms": mean_latency,
            "seconds": round(agent.seconds, 1),
            "stopped_early": agent.stopped_early,
        },
        "trajectory": [
            {
                "generation": record.get("generation"),
                "cycle": record.get("cycle"),
                "score": _finite(record.get("elite_mean_score")),
                "elite": int(record.get("n_elite", 0) or 0),
            }
            for record in records
        ],
        "generations": [_agent_generation_digest(record) for record in records],
        "recent_operations": list(reversed(self._agent_recent.get(name, ()))),
    }
```

在 module helper 区域加入：

```python
def _agent_generation_digest(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generation": record.get("generation"),
        "cycle": record.get("cycle"),
        "generated": int(record.get("n_generated", 0) or 0),
        "passed": int(record.get("n_passed_checker", 0) or 0),
        "qualified": int(record.get("n_qualified", 0) or 0),
        "elite": int(record.get("n_elite", 0) or 0),
        "elite_mean_score": _finite(record.get("elite_mean_score")),
        "best": record.get("best") or {},
        "reject_counts": record.get("reject_counts") or {},
        "op_counts": record.get("op_counts") or {},
        "llm_calls": int(record.get("llm_calls", 0) or 0),
        "wall_seconds": round(float(record.get("wall_seconds", 0.0) or 0.0), 1),
    }
```

- [ ] **Step 7: 运行 reader 测试并修正真实差异**

Run:

```bash
python -m pytest tests/test_monitor_agent_detail.py -v
```

Expected: 4 tests PASS。若浮点平均值或状态与测试不符，先核对 `_apply_generation` 的顺序语义，不得放宽断言来掩盖 agent 混入问题。

---

### Task 2: 提供 Agent HTTP API

**Files:**
- Create: `tests/test_monitor_server.py`
- Modify: `cogalpha/monitor/server.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 补充 TestClient 开发依赖**

将 `pyproject.toml` 的 dev extra 改为：

```toml
dev = ["pytest>=7.0", "httpx>=0.24"]
```

- [ ] **Step 2: 写 Agent API 的失败测试**

写入 `tests/test_monitor_server.py`：

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from cogalpha.monitor.server import build_app


def test_agent_endpoint_returns_known_agent(run_dir):
    with TestClient(build_app(run_dir)) as client:
        response = client.get("/api/agent/AgentMarketCycle")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "AgentMarketCycle"
    assert payload["selected"] is False
    assert payload["recent_operations"] == []


def test_agent_endpoint_returns_404_for_unknown_agent(run_dir):
    with TestClient(build_app(run_dir)) as client:
        response = client.get("/api/agent/AgentDoesNotExist")

    assert response.status_code == 404
    assert response.json()["detail"] == "agent AgentDoesNotExist not found"
```

- [ ] **Step 3: 运行测试并确认路由尚不存在**

Run:

```bash
python -m pytest tests/test_monitor_server.py -v
```

Expected: FAIL，两个请求当前均返回 404，已知 agent 用例不满足 200。

- [ ] **Step 4: 新增 API 路由**

在 `server.py` 的 `/api/call/{seq}` 之前加入：

```python
@app.get("/api/agent/{name}")
def agent(name: str) -> Any:
    """One known hierarchy agent: identity, aggregates and recent activity."""
    reader.poll()
    record = reader.agent_detail(name)
    if record is None:
        raise HTTPException(404, f"agent {name} not found")
    return JSONResponse(record)
```

同时在文件顶部 endpoint 文档列表加入：

```text
GET /api/agent/{name}       one agent's identity, metrics and recent operations
```

- [ ] **Step 5: 运行 API 与 reader 测试**

Run:

```bash
python -m pytest tests/test_monitor_agent_detail.py tests/test_monitor_server.py -v
```

Expected: 6 tests PASS。

---

### Task 3: 建立 Material 3 中文总览与可点击 Agent 卡片

**Files:**
- Create: `tests/test_monitor_ui.py`
- Modify: `cogalpha/monitor/static/index.html`
- Modify: `.gitignore`

- [ ] **Step 1: 忽略视觉讨论临时目录**

在 `.gitignore` 末尾加入：

```gitignore

# --- local design scratch --------------------------------------------------
.superpowers/
```

- [ ] **Step 2: 写 Material 页面结构的失败测试**

写入 `tests/test_monitor_ui.py`：

```python
from pathlib import Path


INDEX = (
    Path(__file__).parents[1]
    / "cogalpha"
    / "monitor"
    / "static"
    / "index.html"
)


def html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_dashboard_uses_material_light_tokens_and_chinese_shell():
    source = html()
    assert '<html lang="zh-CN">' in source
    assert "--md-primary:#0b57d0" in source
    assert "--md-surface:#ffffff" in source
    assert "--md-background:#f8fafd" in source
    assert ">运行总览<" in source
    assert ">Agent 层级<" in source
    assert ">质量漏斗<" in source
    assert ">近期活动<" in source


def test_all_agents_are_rendered_as_accessible_buttons():
    source = html()
    assert 'class="agent-card' in source
    assert 'role="button"' in source
    assert 'tabindex="0"' in source
    assert 'data-agent="${esc(a.name)}"' in source
    assert 'if (!a.selected) m = "本次未选中"' in source


def test_dashboard_has_agent_detail_dialog_structure():
    source = html()
    assert 'id="agent-panel"' in source
    assert 'role="dialog"' in source
    assert 'aria-modal="true"' in source
    assert 'id="agent-tabs"' in source
    assert 'data-tab="overview"' in source
    assert 'data-tab="activity"' in source
    assert 'data-tab="generations"' in source
```

- [ ] **Step 3: 运行静态测试并确认旧页面失败**

Run:

```bash
python -m pytest tests/test_monitor_ui.py -v
```

Expected: 3 tests FAIL，因为旧页面是英文暗色 terminal 风格，且无新的 dialog/tab 结构。

- [ ] **Step 4: 重构页面骨架和 Material tokens**

保留文件开头“单文件、无 CDN、SSE”架构说明，替换 `<html>`、颜色 token 和主要结构。token 必须至少包含：

```css
:root {
  --md-primary:#0b57d0;
  --md-on-primary:#ffffff;
  --md-primary-container:#d3e3fd;
  --md-on-primary-container:#041e49;
  --md-background:#f8fafd;
  --md-surface:#ffffff;
  --md-surface-variant:#eef3f8;
  --md-outline:#c7cdd5;
  --md-outline-variant:#e1e5eb;
  --md-text:#202124;
  --md-text-secondary:#5f6368;
  --md-success:#137333;
  --md-success-container:#e6f4ea;
  --md-warning:#7c5800;
  --md-warning-container:#fef7e0;
  --md-error:#b3261e;
  --md-error-container:#f9dedc;
  --market-up:#d93025;
  --market-down:#188038;
  --radius-card:16px;
  --radius-control:12px;
  --shadow:0 1px 2px rgba(60,64,67,.15),0 2px 6px rgba(60,64,67,.08);
  --sans:"Google Sans",Roboto,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
```

页面主要结构改为：

```html
<body>
  <header class="app-bar">
    <div class="product-mark" aria-hidden="true">C</div>
    <div class="app-title"><strong>CogAlpha</strong><span>智能因子研究监控</span></div>
    <span id="status" class="status-chip">正在连接</span>
    <div class="run-meta"><span id="runname"></span><span id="cfg"></span></div>
    <div class="header-actions"><span id="updated-at">等待数据</span></div>
  </header>
  <main>
    <section id="warnbox" class="warning-stack" hidden></section>
    <section class="hero-section" aria-labelledby="overview-title">
      <div class="section-heading"><div><p class="eyebrow">RUN OVERVIEW</p><h1 id="overview-title">运行总览</h1></div></div>
      <div class="metric-grid" id="stats"></div>
    </section>
    <section class="card full" aria-labelledby="agents-title">
      <div class="card-heading"><div><h2 id="agents-title">Agent 层级</h2><p id="agentnote"></p></div></div>
      <div id="matrix" class="agent-matrix"></div>
    </section>
    <div class="dashboard-grid">
      <section class="card"><div class="card-heading"><h2>质量漏斗</h2><p id="funnelnote"></p></div><div id="funnel"></div></section>
      <section class="card"><div class="card-heading"><h2>Elite 趋势</h2><p>每代平均分，分隔线表示 Agent 切换</p></div><svg id="traj" viewBox="0 0 560 180"></svg><div id="plateau"></div></section>
      <section class="card"><div class="card-heading"><h2>LLM 成本</h2></div><div class="table-wrap"><table><thead><tr><th>角色</th><th>调用</th><th>Tokens</th><th>占比</th><th>平均耗时</th></tr></thead><tbody id="roles"></tbody></table></div></section>
      <section class="card"><div class="card-heading"><h2>近期活动</h2><p>点击查看 prompt 与 response</p></div><div class="table-wrap"><table><thead><tr><th>#</th><th>角色</th><th>Agent</th><th>代</th><th>Tokens</th><th>耗时</th></tr></thead><tbody id="calls"></tbody></table></div></section>
    </div>
    <section class="card full" id="candsec" hidden><div class="card-heading"><h2>候选因子</h2></div><div class="table-wrap"><table><tbody id="cands"></tbody></table></div></section>
  </main>
  <div id="scrim" class="scrim"></div>
  <aside id="agent-panel" class="agent-panel" role="dialog" aria-modal="true" aria-labelledby="agent-panel-title" aria-hidden="true">
    <div class="agent-panel-head"><div><p id="agent-panel-level" class="eyebrow"></p><h2 id="agent-panel-title">Agent 详情</h2><p id="agent-panel-subtitle"></p></div><button id="agent-panel-close" class="icon-button" aria-label="关闭 Agent 详情">×</button></div>
    <nav id="agent-tabs" class="tabs" aria-label="Agent 详情栏目"><button data-tab="overview" class="active">概览</button><button data-tab="activity">活动</button><button data-tab="generations">每代结果</button></nav>
    <div id="agent-panel-body" class="agent-panel-body"></div>
  </aside>
  <div id="detail-layer" class="detail-layer" aria-hidden="true"></div>
</body>
```

候选表头可保留现有指标列；不得删除现有 `cands` 渲染目标。

- [ ] **Step 5: 把 Agent matrix 改为 21 个都可交互**

新增中文层级名称：

```javascript
const LEVELS_ZH = {
  1:"I 市场结构与周期",2:"II 极端风险与脆弱性",3:"III 量价动力",
  4:"IV 价格与波动行为",5:"V 多尺度复杂性",6:"VI 稳定性与状态门控",
  7:"VII 几何形态与融合"
};
```

`renderMatrix` 中每个 agent 都输出统一的可交互卡片；不要再用 `a.selected ? data-agent : ""` 禁用未选中项：

```javascript
const cells = byLevel[lv].map((a) => {
  const state = a.selected ? a.status : "unselected";
  let meta;
  if (!a.selected) meta = "本次未选中";
  else if (a.status === "queued") meta = "排队中";
  else meta = `第 ${a.generations} 代 · Qualified ${a.n_qualified} · Elite ${a.n_elite}`;
  return `<div class="agent-card ${state}" role="button" tabindex="0" data-agent="${esc(a.name)}" aria-label="查看 ${esc(a.name)} 详情">
    <div class="agent-card-top"><span class="agent-status-dot" aria-hidden="true"></span><strong>${esc(a.name.replace(/^Agent/, ""))}</strong><span class="agent-state-label">${stateLabel(state)}</span></div>
    <p>${meta}</p><div class="agent-progress"><span style="width:${agentProgress(a)}%"></span></div>
  </div>`;
}).join("");
```

补充并使用这些 helper：

```javascript
function stateLabel(state) {
  return {running:"运行中",done:"已完成",queued:"排队中",unselected:"未选中"}[state] || state;
}
function agentProgress(a) {
  if (!a.selected) return 0;
  const planned = Math.max(1, +(LAST?.config?.generations || 0));
  return Math.min(100, Math.round((a.generations || 0) / planned * 100));
}
function bindAgentCards() {
  document.querySelectorAll("[data-agent]").forEach((el) => {
    el.onclick = () => showAgent(el.dataset.agent, el);
    el.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showAgent(el.dataset.agent, el);
      }
    };
  });
}
```

在 `renderMatrix` 写入 DOM 后调用 `bindAgentCards()`。

- [ ] **Step 6: 运行静态 UI 测试**

Run:

```bash
python -m pytest tests/test_monitor_ui.py -v
```

Expected: 3 tests PASS。此时详情内容尚未实现，但结构必须存在。

---

### Task 4: 实现 Agent 面板、标签页、实时刷新与深层详情

**Files:**
- Modify: `tests/test_monitor_ui.py`
- Modify: `cogalpha/monitor/static/index.html`

- [ ] **Step 1: 写详情交互源代码保护测试**

追加到 `tests/test_monitor_ui.py`：

```python
def test_agent_panel_fetches_detail_and_supports_all_tabs():
    source = html()
    assert "async function loadAgentDetail(name)" in source
    assert "`/api/agent/${encodeURIComponent(name)}`" in source
    assert "function renderAgentOverview(detail)" in source
    assert "function renderAgentActivity(detail)" in source
    assert "function renderAgentGenerations(detail)" in source
    assert "function renderAgentEmpty(detail)" in source


def test_agent_panel_has_keyboard_and_change_triggered_refresh():
    source = html()
    assert 'event.key === "Escape"' in source
    assert "AGENT_REFRESH_MIN_MS = 2000" in source
    assert "function agentRevision(snapshot, name)" in source
    assert "scheduleAgentRefresh()" in source


def test_archive_text_is_escaped_before_rendering():
    source = html()
    assert "esc(detail.focus)" in source
    assert "esc(detail.probe)" in source
    assert "esc(op.role)" in source
    assert "esc(row.reject_reason)" in source
```

- [ ] **Step 2: 运行新增测试并确认交互尚未实现**

Run:

```bash
python -m pytest tests/test_monitor_ui.py -v
```

Expected: 新增的 3 tests FAIL。

- [ ] **Step 3: 加入 Agent 详情状态机与打开/关闭行为**

在现有 `LAST` 之后定义：

```javascript
const AGENT_REFRESH_MIN_MS = 2000;
let AGENT_VIEW = {
  name:null,
  tab:"overview",
  detail:null,
  trigger:null,
  loading:false,
  lastLoadedAt:0,
  revision:"",
  timer:null
};

function showAgent(name, trigger = null) {
  AGENT_VIEW.name = name;
  AGENT_VIEW.tab = "overview";
  AGENT_VIEW.trigger = trigger || document.querySelector(`[data-agent="${CSS.escape(name)}"]`);
  $("agent-panel").classList.add("open");
  $("agent-panel").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");
  document.body.classList.add("panel-open");
  $("agent-panel-close").focus();
  renderAgentLoading();
  loadAgentDetail(name);
}

function closeAgentPanel() {
  $("agent-panel").classList.remove("open");
  $("agent-panel").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
  document.body.classList.remove("panel-open");
  clearTimeout(AGENT_VIEW.timer);
  const trigger = AGENT_VIEW.trigger;
  AGENT_VIEW = {name:null,tab:"overview",detail:null,trigger:null,loading:false,lastLoadedAt:0,revision:"",timer:null};
  if (trigger) trigger.focus();
}

function renderAgentLoading() {
  $("agent-panel-body").innerHTML = '<div class="skeleton-stack"><div></div><div></div><div></div></div>';
}

async function loadAgentDetail(name) {
  if (AGENT_VIEW.loading || AGENT_VIEW.name !== name) return;
  AGENT_VIEW.loading = true;
  try {
    const response = await fetch(`/api/agent/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    if (AGENT_VIEW.name !== name) return;
    AGENT_VIEW.detail = detail;
    AGENT_VIEW.lastLoadedAt = Date.now();
    AGENT_VIEW.revision = agentRevision(LAST, name);
    renderAgentDetail();
  } catch (error) {
    if (AGENT_VIEW.name === name) {
      $("agent-panel-body").innerHTML = `<div class="error-state"><strong>Agent 详情加载失败</strong><p>${esc(error.message)}</p><button onclick="loadAgentDetail('${esc(name)}')">重试</button></div>`;
    }
  } finally {
    AGENT_VIEW.loading = false;
  }
}
```

绑定关闭、遮罩、Escape 和 tabs：

```javascript
$("agent-panel-close").onclick = closeAgentPanel;
$("scrim").onclick = closeAgentPanel;
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && AGENT_VIEW.name) closeAgentPanel();
});
$("agent-tabs").querySelectorAll("[data-tab]").forEach((button) => {
  button.onclick = () => {
    AGENT_VIEW.tab = button.dataset.tab;
    renderAgentDetail();
  };
});
```

- [ ] **Step 4: 实现中文职责、空状态和三个标签页渲染**

定义 21 个 agent 的中文简述。键必须与 `HIERARCHY` 名称一一对应：

```javascript
const AGENT_SUMMARIES_ZH = {
  AgentMarketCycle:"识别长期趋势、市场阶段和周期状态转换。",
  AgentVolatilityRegime:"识别波动率状态及其切换。",
  AgentTailRisk:"衡量左尾风险暴露和损失分布形态。",
  AgentCrashPredictor:"寻找市场崩跌前的压力累积信号。",
  AgentLiquidity:"刻画流动性条件与单位成交量价格冲击。",
  AgentOrderImbalance:"从 OHLCV 几何推断买卖压力失衡。",
  AgentPriceVolumeCoherence:"衡量价格走势与成交量行为是否一致。",
  AgentVolumeStructure:"研究交易参与度、集中度和吸收现象。",
  AgentDailyTrend:"分析方向持续性和多日动量强度。",
  AgentReversal:"识别短期过度反应及其均值回归。",
  AgentRangeVol:"研究振幅波动和收缩—扩张周期。",
  AgentLagResponse:"识别波动、成交量与收益之间的滞后反馈。",
  AgentVolAsymmetry:"比较上涨与下跌阶段的非对称波动。",
  AgentDrawdown:"刻画回撤深度、持续时间与恢复路径。",
  AgentFractal:"衡量多尺度粗糙度和长期记忆。",
  AgentRegimeGating:"构造随市场状态变化的自适应信号门控。",
  AgentStability:"评估收益与衍生信号的时间稳定性。",
  AgentBarShape:"将 K 线实体、影线和对称性编码为连续特征。",
  AgentCreative:"探索非线性变换、重参数化与软门控。",
  AgentComposite:"融合独立因子，强调协同与正交性。",
  AgentHerding:"识别群体行为、拥挤交易与同步运动。"
};
```

实现统一入口和空状态：

```javascript
function renderAgentDetail() {
  const detail = AGENT_VIEW.detail;
  if (!detail) return;
  $("agent-panel-level").textContent = `LEVEL ${detail.level} · ${LEVELS_ZH[detail.level] || detail.layer}`;
  $("agent-panel-title").textContent = detail.display_name;
  $("agent-panel-subtitle").textContent = AGENT_SUMMARIES_ZH[detail.name] || detail.focus;
  $("agent-tabs").querySelectorAll("[data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === AGENT_VIEW.tab));
  if (!detail.selected || (detail.summary.generations === 0 && detail.recent_operations.length === 0)) {
    renderAgentEmpty(detail);
    return;
  }
  if (AGENT_VIEW.tab === "activity") renderAgentActivity(detail);
  else if (AGENT_VIEW.tab === "generations") renderAgentGenerations(detail);
  else renderAgentOverview(detail);
}

function renderAgentEmpty(detail) {
  const message = !detail.selected
    ? "本次运行未选中，因此暂无运行数据。"
    : detail.status === "queued"
      ? "已选中，正在等待前序 Agent 完成。"
      : "Agent 已启动，正在等待首条运行记录。";
  $("agent-panel-body").innerHTML = `<div class="empty-state"><div class="empty-icon">◎</div><h3>${esc(stateLabel(detail.selected ? detail.status : "unselected"))}</h3><p>${esc(message)}</p><details><summary>查看研究定义</summary><h4>Focus</h4><p>${esc(detail.focus)}</p><h4>Probe</h4><p>${esc(detail.probe)}</p></details></div>`;
}
```

实现概览指标、轨迹和研究定义：

```javascript
function renderAgentOverview(detail) {
  const s = detail.summary;
  const metrics = [
    ["Generated", s.generated], ["Passed", s.passed],
    ["Qualified", s.qualified], ["Elite", s.elite],
    ["LLM Calls", s.llm_calls], ["Tokens", K(s.llm_tokens)],
    ["Best RankIC", s.best_rank_ic == null ? "–" : sgn(s.best_rank_ic)],
    ["耗时", dur(s.seconds)]
  ].map(([label, value]) => `<div class="detail-metric"><small>${label}</small><strong>${value}</strong></div>`).join("");
  const points = detail.trajectory.filter((point) => point.score != null);
  const chart = miniTrajectory(points);
  $("agent-panel-body").innerHTML = `<div class="detail-metric-grid">${metrics}</div>
    <section class="detail-section"><div class="panel-section-heading"><div><h3>Elite 轨迹</h3><p>该 Agent 每代 Elite 平均分</p></div></div>${chart}</section>
    <details class="research-definition"><summary>查看研究定义</summary><h4>Focus</h4><p>${esc(detail.focus)}</p><h4>Probe</h4><p>${esc(detail.probe)}</p></details>`;
}

function miniTrajectory(points) {
  if (!points.length) return '<div class="empty-state compact"><p>尚无 Elite 分数。</p></div>';
  const width = 560, height = 150, pad = 18;
  const values = points.map((point) => +point.score);
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi === lo) { hi += 0.001; lo -= 0.001; }
  const xy = points.map((point, index) => {
    const x = pad + index * (width - 2 * pad) / Math.max(1, points.length - 1);
    const y = height - pad - (+point.score - lo) / (hi - lo) * (height - 2 * pad);
    return [x, y];
  });
  const path = xy.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return `<svg class="mini-trajectory" viewBox="0 0 ${width} ${height}" role="img" aria-label="Elite 分数趋势"><path class="grid-line" d="M${pad},${height / 2}H${width - pad}"></path><path class="trajectory-line" d="${path}"></path>${xy.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="3"></circle>`).join("")}</svg>`;
}
```

实现 `renderAgentActivity(detail)`：

```javascript
function renderAgentActivity(detail) {
  const rows = detail.recent_operations.map((op) => `<button class="timeline-item" onclick="showCall(${Number(op.seq)})">
    <span class="timeline-dot ${esc(op.role)}"></span>
    <span><strong>${esc(op.role)}</strong><small>Generation ${esc(op.generation ?? "–")} · Cycle ${esc(op.cycle ?? "–")} · ${esc(op.mode || "default")}</small></span>
    <span class="timeline-meta">${K(op.tokens)} tokens<br>${esc(op.latency_ms ?? "–")} ms</span>
  </button>`).join("");
  $("agent-panel-body").innerHTML = `<div class="panel-section-heading"><div><h3>最近活动</h3><p>最多保留该 Agent 最近 200 条调用摘要</p></div><span class="count-chip">${detail.recent_operations.length}</span></div>${rows || '<div class="empty-state compact"><p>暂无 LLM 调用记录。</p></div>'}`;
}
```

实现每代结果：

```javascript
function renderAgentGenerations(detail) {
  const rows = detail.generations.slice().reverse().map((row) => `<button class="generation-row" onclick="showGeneration('${esc(detail.name)}', ${Number(row.generation)})">
    <span><strong>Generation ${esc(row.generation)}</strong><small>Cycle ${esc(row.cycle)} · ${dur(row.wall_seconds)}</small></span>
    <span><small>Generated</small><strong>${row.generated}</strong></span>
    <span><small>Passed</small><strong>${row.passed}</strong></span>
    <span><small>Qualified</small><strong>${row.qualified}</strong></span>
    <span><small>Elite</small><strong>${row.elite}</strong></span>
    <span><small>Score</small><strong>${row.elite_mean_score == null ? "–" : num(row.elite_mean_score)}</strong></span>
  </button>`).join("");
  $("agent-panel-body").innerHTML = `<div class="panel-section-heading"><div><h3>每代结果</h3><p>点击一代查看产生的 alpha 与淘汰原因</p></div><span class="count-chip">${detail.generations.length}</span></div>${rows || '<div class="empty-state compact"><p>暂无 generation 记录。</p></div>'}`;
}
```

- [ ] **Step 5: 让深层详情在同一详情层中可返回**

保留现有 `showCall`、`showGeneration`、`showAlpha`、`showCandidate` 的数据请求与内容模板，但把原 `openDrawer` 改为 `openDetailLayer(title, html)`：

```javascript
function openDetailLayer(title, html) {
  $("detail-layer").innerHTML = `<div class="detail-layer-head"><button id="detail-back" class="text-button">← 返回 Agent</button><strong>${title}</strong></div><div class="detail-layer-body">${html}</div>`;
  $("detail-layer").classList.add("open");
  $("detail-layer").setAttribute("aria-hidden", "false");
  $("detail-back").onclick = closeDetailLayer;
}

function closeDetailLayer() {
  $("detail-layer").classList.remove("open");
  $("detail-layer").setAttribute("aria-hidden", "true");
  $("detail-layer").innerHTML = "";
}
```

`showCall`、`showGeneration`、`showAlpha`、`showCandidate` 中所有原 `openDrawer(...)` 调用替换为 `openDetailLayer(...)`。关闭 Agent 面板时先调用 `closeDetailLayer()`。

- [ ] **Step 6: 实现变化触发的详情刷新**

加入：

```javascript
function agentRevision(snapshot, name) {
  const agent = (snapshot?.agents || []).find((item) => item.name === name);
  if (!agent) return "";
  return [agent.status, agent.generations, agent.llm_calls, agent.n_generated, agent.n_qualified, agent.n_elite].join(":");
}

function scheduleAgentRefresh() {
  if (!AGENT_VIEW.name || AGENT_VIEW.loading) return;
  const next = agentRevision(LAST, AGENT_VIEW.name);
  if (!next || next === AGENT_VIEW.revision) return;
  const wait = Math.max(0, AGENT_REFRESH_MIN_MS - (Date.now() - AGENT_VIEW.lastLoadedAt));
  clearTimeout(AGENT_VIEW.timer);
  AGENT_VIEW.timer = setTimeout(() => loadAgentDetail(AGENT_VIEW.name), wait);
}
```

在全局 `render(s)` 完成各面板渲染后调用 `scheduleAgentRefresh()`。如果 matrix 因 SSE 重绘，不得覆盖 `AGENT_VIEW.trigger`；关闭时如果旧节点已被替换，使用 `document.querySelector` 按 `data-agent` 重新寻找并聚焦。

- [ ] **Step 7: 完成响应式和减少动画偏好**

CSS 必须包含：

```css
.agent-panel{position:fixed;top:0;right:0;width:clamp(620px,60vw,980px);height:100dvh;transform:translateX(105%);transition:transform .22s cubic-bezier(.2,0,0,1);z-index:40;background:var(--md-surface);box-shadow:-12px 0 32px rgba(60,64,67,.18);display:flex;flex-direction:column}
.agent-panel.open{transform:none}.agent-panel-body{padding:20px 24px 40px;overflow:auto;flex:1}.detail-layer{position:absolute;inset:0;background:var(--md-surface);transform:translateX(102%);transition:transform .2s;z-index:2}.detail-layer.open{transform:none}
@media(max-width:760px){.agent-panel{width:100vw}.dashboard-grid{grid-template-columns:1fr}.app-bar{position:static}.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition-duration:.01ms!important;animation-duration:.01ms!important;animation-iteration-count:1!important}}
```

- [ ] **Step 8: 运行全部自动测试**

Run:

```bash
python -m pytest -v
```

Expected: 所有测试 PASS，0 failures。

---

### Task 5: 更新文档并用真实归档做视觉与交互验收

**Files:**
- Modify: `README.md`
- Verify: `cogalpha/monitor/static/index.html`
- Verify: `runs/20260810-123307-csi300/`
- Verify: `runs/20260810-084359-csi300/`

- [ ] **Step 1: 更新 README 的 dashboard 描述**

将 `## What the dashboard shows` 的开头两段改为：

```markdown
## What the dashboard shows

The Material 3 dashboard presents the 21-agent hierarchy, run progress, the current
quality-checker funnel, elite trajectory, LLM cost by role, recent activity, and saved
candidates. The interface is Chinese-first while agent names and quantitative fields
remain in English.

Every agent is clickable, including agents not selected for the current run. The
right-side detail panel shows the agent's scope, status, aggregate yield, elite
trajectory, up to 200 recent LLM-operation summaries, and generation-by-generation
results. From there, drill down into a generation, alpha source and checks, or an
individual LLM prompt and response. Unselected and queued agents show an explicit
empty state instead of misleading zero metrics.
```

保留后面的三类自动 warning 和绑定安全说明。

- [ ] **Step 2: 启动完成归档 monitor**

Run:

```bash
python -m cogalpha.cli monitor --run runs/20260810-123307-csi300 --port 8080
```

Expected: 控制台显示 `cogalpha monitor | http://127.0.0.1:8080`，无 traceback。

- [ ] **Step 3: 使用浏览器检查桌面总览**

打开 `http://127.0.0.1:8080`，检查：

- 页面是浅色 Material 3，不存在旧暗色 terminal 大面积残留。
- 顶部与卡片中文可读，技术字段保持英文。
- 核心指标、21 个 agent、漏斗、趋势、成本、活动和候选结果均有内容。
- 1920×1080 和约 1280px 宽度下没有水平滚动或文本遮挡。
- 正负金融指标仍保持红涨绿跌；运行状态不复用金融涨跌颜色。

- [ ] **Step 4: 检查四类 Agent 状态和详情链路**

在完成归档中至少验证：

- 点击已完成 agent：概览、活动、每代结果均有数据。
- 点击未选中 agent：显示中文职责、英文 research definition 和明确空状态。
- 活动记录点击后能查看 prompt/response，并能返回 Agent。
- 某代点击后能进入 alpha 列表；alpha 点击后显示代码、checks、fitness 和 lineage。
- 点击遮罩、关闭按钮和按 Escape 均可关闭；关闭后焦点回到原卡片。

- [ ] **Step 5: 检查不完整归档和窄屏**

停止前一个 monitor，然后运行：

```bash
python -m cogalpha.cli monitor --run runs/20260810-084359-csi300 --port 8080
```

Expected: 缺少完整 alpha/summary 数据时页面仍能打开，详情显示可解释的空状态。

在浏览器约 390×844 viewport 检查：

- Agent 详情变为全屏。
- 指标卡为两列或单列，不溢出。
- tabs、表格和详情内容可滚动且按钮可点击。

- [ ] **Step 6: 运行最终验证命令**

Run:

```bash
python -m pytest -v
python -m compileall -q cogalpha tests
```

Expected: pytest 0 failures；compileall 退出码 0 且无输出。

- [ ] **Step 7: 检查变更范围**

Run:

```bash
find tests docs/superpowers -type f -maxdepth 4 | sort
```

手工确认最终实现只修改 file map 中列出的 monitor、测试、依赖、ignore 和 README 文件；不修改搜索、进化、质量检查和归档写入逻辑。

---

## Final acceptance checklist

- [ ] 全部 21 个 agent 均可点击，未选中 agent 不再禁用。
- [ ] Agent API 不返回 prompt/response 正文。
- [ ] 每个 agent 的活动摘要固定上限为 200，累计 calls/tokens 不受上限影响。
- [ ] 打开的 Agent 面板随 SSE 摘要变化进行不短于 2 秒的防抖刷新。
- [ ] 概览、活动、每代结果和深层详情均可导航和返回。
- [ ] 中文优先与 Material 3 浅色视觉符合已批准 mockup。
- [ ] 完成归档、不完整归档、桌面与窄屏均通过手工验收。
- [ ] 全部 pytest 与 compileall 验证通过。
