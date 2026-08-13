"""Static contract tests for the dependency-free monitor dashboard."""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from html.parser import HTMLParser
from pathlib import Path


HTML_PATH = (
    Path(__file__).resolve().parents[1]
    / "cogalpha"
    / "monitor"
    / "static"
    / "index.html"
)
HTML = HTML_PATH.read_text(encoding="utf-8")
STYLE = HTML.split("<style>", 1)[1].split("</style>", 1)[0]
SCRIPT = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
SCRIPT_WITHOUT_CONNECT = re.sub(r"\nconnect\(\);\s*$", "", SCRIPT)


class _MarkupContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.data_tabs: list[str] = []
        self.script_sources: list[str] = []
        self.link_hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if attributes.get("id") is not None:
            self.ids.append(attributes["id"] or "")
        if attributes.get("data-tab") is not None:
            self.data_tabs.append(attributes["data-tab"] or "")
        if tag == "script" and attributes.get("src") is not None:
            self.script_sources.append(attributes["src"] or "")
        if tag == "link" and attributes.get("href") is not None:
            self.link_hrefs.append(attributes["href"] or "")


class _GeneratedMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.tags.append(tag)
        self.attributes.extend(name for name, _value in attrs)


def _parse_markup() -> _MarkupContractParser:
    parser = _MarkupContractParser()
    parser.feed(HTML)
    return parser


def _css_rule(selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", STYLE)
    assert match is not None, f"missing CSS rule for {selector}"
    return match.group(1)


NODE_DOM_STUB = r"""
class ClassList {
  constructor() { this.values = new Set(); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  contains(value) { return this.values.has(value); }
  toggle(value, force) {
    if (force === undefined) force = !this.values.has(value);
    if (force) this.values.add(value); else this.values.delete(value);
    return force;
  }
}
class Element {
  constructor(id = "") {
    this.id = id;
    this.innerHTML = "";
    this.textContent = "";
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.classList = new ClassList();
    this.hidden = false;
    this.inert = false;
    this.disabled = false;
    this.tabIndex = 0;
    this.parentElement = null;
    this.rects = [{}];
    this.computedStyle = {};
    this.childrenForQuery = [];
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "hidden") this.hidden = true;
    if (name === "inert") this.inert = true;
    if (name === "tabindex") this.tabIndex = Number(value);
  }
  removeAttribute(name) {
    delete this.attributes[name];
    if (name === "hidden") this.hidden = false;
    if (name === "inert") this.inert = false;
  }
  getAttribute(name) { return this.attributes[name] ?? null; }
  focus() { document.activeElement = this; this.focused = true; }
  querySelectorAll() { return this.childrenForQuery; }
  querySelector() { return this.childrenForQuery[0] || null; }
  closest() { return this; }
  contains(element) {
    return this === element || this.childrenForQuery.some((child) =>
      child === element || (child.contains && child.contains(element)));
  }
  getClientRects() { return this.rects; }
}
const elements = new Map();
const document = {
  activeElement: null,
  listeners: {},
  detached: new Set(),
  documentElement: new Element("documentElement"),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  },
  addEventListener(type, listener) { this.listeners[type] = listener; },
  contains(element) { return !this.detached.has(element); },
  querySelectorAll() { return []; },
};
globalThis.document = document;
globalThis.window = globalThis;
globalThis.getComputedStyle = (element) => ({
  ...(element && element.computedStyle ? element.computedStyle : {}),
  getPropertyValue(name) {
    return ({
      "--md-primary": "#0b57d0",
      "--md-outline": "#c7cdd5",
      "--md-outline-variant": "#e1e5eb",
      "--md-text-secondary": "#5f6368",
    })[name] || "";
  },
});
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {};
globalThis.setTimeout = () => 1;
globalThis.clearTimeout = () => {};
globalThis.fetch = async () => ({ok: true, json: async () => ({}), text: async () => ""});
globalThis.EventSource = class {
  constructor() { globalThis.lastEventSource = this; this.closed = false; }
  close() { this.closed = true; }
};
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const response = (body, ok = true) => ({
  ok,
  json: async () => body,
  text: async () => typeof body === "string" ? body : JSON.stringify(body),
});
"""


def _run_dashboard_node(tmp_path: Path, assertions: str) -> str:
    script_path = tmp_path / "monitor-ui-contract.js"
    script_path.write_text(
        NODE_DOM_STUB + SCRIPT_WITHOUT_CONNECT + textwrap.dedent(assertions),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_dashboard_uses_chinese_material_three_shell() -> None:
    assert '<html lang="zh-CN">' in HTML
    assert "--md-primary:#0b57d0" in HTML
    assert "--md-surface:#ffffff" in HTML
    assert "--md-background:#f8fafd" in HTML

    for label in ("运行总览", "Agent 层级", "质量漏斗", "近期活动"):
        assert label in HTML


def test_every_agent_card_is_keyboard_interactive() -> None:
    render_matrix = HTML.split("function renderMatrix", 1)[1].split(
        "function renderFunnel", 1
    )[0]

    assert 'class="agent-card ${state}"' in render_matrix
    assert 'role="button"' in render_matrix
    assert 'tabindex="0"' in render_matrix
    assert 'data-agent="${esc(a.name)}"' in render_matrix
    assert "a.selected ? ` data-agent=" not in render_matrix
    assert "本次未选中" in render_matrix
    assert "bindAgentCards" in render_matrix


def test_agent_dialog_exposes_tabs_for_follow_up_detail_work() -> None:
    assert re.search(
        r'<aside\b[^>]*id="agent-panel"[^>]*role="dialog"'
        r'[^>]*aria-modal="true"[^>]*aria-hidden="true"',
        HTML,
    )
    assert 'id="agent-panel-head" class="panel-head"' in HTML
    assert 'id="agent-tabs"' in HTML
    assert 'id="agent-panel-body"' in HTML

    tabs = _parse_markup().data_tabs
    assert len(tabs) == 3
    assert set(tabs) == {"overview", "activity", "generations"}


def test_all_required_render_targets_and_functions_remain_available() -> None:
    for element_id in (
        "stats",
        "matrix",
        "funnel",
        "traj",
        "plateau",
        "roles",
        "calls",
        "candsec",
        "cands",
        "warnbox",
        "status",
        "runname",
        "cfg",
    ):
        assert f'id="{element_id}"' in HTML

    for function_name in (
        "renderFunnel",
        "renderTraj",
        "renderRoles",
        "renderCalls",
        "renderCands",
        "showCall",
        "showGeneration",
        "showAlpha",
        "showCandidate",
        "connect",
        "poll",
    ):
        assert f"function {function_name}" in HTML


def test_chinese_level_labels_match_the_seven_layer_model() -> None:
    levels_match = re.search(r"const LEVELS_ZH = \{(.*?)\};", HTML, re.DOTALL)
    assert levels_match is not None
    entries = re.findall(r'^\s*(\d+):\s*"([^"]+)"\s*,?\s*$', levels_match.group(1), re.MULTILINE)

    assert len(entries) == 7
    assert [int(level) for level, _label in entries] == list(range(1, 8))
    assert {int(level): label for level, label in entries} == {
        1: "I 市场结构与周期",
        2: "II 极端风险与脆弱性",
        3: "III 量价动力",
        4: "IV 价格与波动行为",
        5: "V 多尺度复杂性",
        6: "VI 稳定性与状态门控",
        7: "VII 几何形态与融合",
    }


def test_agent_chinese_summaries_match_all_21_canonical_responsibilities() -> None:
    summaries_match = re.search(
        r"const AGENT_SUMMARIES_ZH = \{(.*?)\};", HTML, re.DOTALL
    )
    assert summaries_match is not None
    entries = re.findall(
        r'^\s*([A-Za-z0-9_]+):\s*"([^"]+)"\s*,?\s*$',
        summaries_match.group(1),
        re.MULTILINE,
    )

    assert dict(entries) == {
        "AgentMarketCycle": "识别长期趋势、市场阶段和周期状态转换。",
        "AgentVolatilityRegime": "识别波动率状态及其切换。",
        "AgentTailRisk": "衡量左尾风险暴露和损失分布形态。",
        "AgentCrashPredictor": "寻找市场崩跌前的压力累积信号。",
        "AgentLiquidity": "刻画流动性条件与单位成交量价格冲击。",
        "AgentOrderImbalance": "从 OHLCV 几何推断买卖压力失衡。",
        "AgentPriceVolumeCoherence": "衡量价格走势与成交量行为是否一致。",
        "AgentVolumeStructure": "研究交易参与度、集中度和吸收现象。",
        "AgentDailyTrend": "分析方向持续性和多日动量强度。",
        "AgentReversal": "识别短期过度反应及其均值回归。",
        "AgentRangeVol": "研究振幅波动和收缩—扩张周期。",
        "AgentLagResponse": "识别波动、成交量与收益之间的滞后反馈。",
        "AgentVolAsymmetry": "比较上涨与下跌阶段的非对称波动。",
        "AgentDrawdown": "刻画回撤深度、持续时间与恢复路径。",
        "AgentFractal": "衡量多尺度粗糙度和长期记忆。",
        "AgentRegimeGating": "构造随市场状态变化的自适应信号门控。",
        "AgentStability": "评估收益与衍生信号的时间稳定性。",
        "AgentBarShape": "将 K 线实体、影线和对称性编码为连续特征。",
        "AgentCreative": "探索非线性变换、重参数化与软门控。",
        "AgentComposite": "融合独立因子，强调协同与正交性。",
        "AgentHerding": "识别群体行为、拥挤交易与同步运动。",
    }
    assert len(entries) == 21


def test_agent_empty_states_distinguish_unselected_queued_and_started(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        const attack = `"><img src=x onerror=globalThis.pwned=1>`;
        const base = {
          name: "AgentMarketCycle", focus: attack, probe: attack,
          summary: {generations: 0, llm_calls: 0}, recent_operations: [],
        };
        const unselected = renderAgentEmpty({...base, selected: false, status: "queued"});
        assert(unselected.includes("本次运行未选中，因此暂无运行数据"),
          "unselected state is ambiguous");
        assert(unselected.includes(AGENT_SUMMARIES_ZH.AgentMarketCycle),
          "unselected state lacks Chinese responsibility");
        assert(unselected.includes("&lt;img") && !/<img\b/i.test(unselected),
          "research definitions were not escaped");

        const queued = renderAgentEmpty({...base, selected: true, status: "queued"});
        assert(queued.includes("已选中，正在等待前序 Agent 完成"),
          "queued state is ambiguous");
        assert(!queued.includes("agent-summary-grid"),
          "queued state shows a misleading zero metric grid");

        const running = renderAgentEmpty({...base, selected: true, status: "running"});
        assert(running.includes("Agent 已启动，正在等待首条运行记录"),
          "started-empty state is ambiguous");
        assert(!running.includes("agent-summary-grid"),
          "started-empty state shows a misleading zero metric grid");
        assert(!globalThis.pwned, "malicious research definition executed");
        """,
    )


def test_agent_overview_normalizes_metrics_and_trajectory_edges(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        const attack = `"><svg onload=globalThis.pwned=1>`;
        const garbage = `Infinity${attack}`;
        const detail = {
          name: "AgentMarketCycle", selected: true, status: "done",
          current_generation: 3, current_cycle: 2, focus: attack, probe: attack,
          summary: {
            generated: 8, passed: garbage, qualified: 3, elite: 1,
            llm_calls: 7, llm_tokens: garbage, best_rank_ic: -0.25,
            seconds: 65, stopped_early: attack,
          },
          trajectory: [
            {score: garbage, generation: garbage},
            {score: 0.2, generation: 1},
            {score: 0.2, generation: 2},
          ],
          recent_operations: [{seq: 1}],
        };
        const overview = renderAgentOverview(detail);
        for (const label of [
          "Generated", "Passed", "Qualified", "Elite", "LLM Calls",
          "Tokens", "Best RankIC", "耗时",
        ]) assert(overview.includes(label), `missing ${label} summary`);
        assert(overview.includes('class="neg"') && overview.includes("-0.2500"),
          "signed market metric does not use down/green semantics");
        assert(overview.includes("第 3 代") && overview.includes("周期 2"),
          "current run context is missing");
        assert(overview.includes("已完成"), "overview current state is missing");
        assert(overview.includes("提前停止") && overview.includes("&lt;svg"),
          "stopped reason or research definition was not escaped");

        const svgs = [
          miniTrajectory([]),
          miniTrajectory([{score: 0.1}]),
          miniTrajectory([{score: 0.1}, {score: 0.1}, {score: garbage}]),
        ];
        for (const svg of svgs) {
          assert(svg.includes("<svg") && svg.includes("aria-label="),
            "mini trajectory is not an accessible SVG");
          assert(!/NaN|Infinity|undefined/.test(svg),
            "mini trajectory leaked a non-finite value");
        }
        const generated = overview + svgs.join("");
        assert(!/<svg\b[^>]*\sonload=/i.test(generated), "payload created an SVG handler");
        assert(!/NaN|Infinity|undefined/.test(generated), "overview leaked numeric garbage");
        assert(!globalThis.pwned, "malicious overview payload executed");
        """,
    )


def test_agent_mini_trajectory_scales_opposite_finite_number_extremes(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        const svg = miniTrajectory([
          {score: Number.MAX_VALUE, generation: 1},
          {score: -Number.MAX_VALUE, generation: 2},
        ]);
        assert(svg.includes("<svg") && svg.includes("aria-label="),
          "extreme trajectory is not an accessible SVG");
        assert(svg.includes("#0b57d0") && svg.includes("#e1e5eb"),
          "extreme trajectory stopped using CSS theme token values");
        assert(!/NaN|Infinity|undefined/.test(svg),
          "finite extreme scores produced a non-finite SVG value");
        const numericAttributes = Array.from(svg.matchAll(
          /\b(?:x1|x2|y1|y2|cx|cy|r|stroke-width)="([^"]+)"/g
        )).map((match) => Number(match[1]));
        assert(numericAttributes.length > 0 && numericAttributes.every(Number.isFinite),
          "extreme trajectory produced a non-finite coordinate");
        """,
    )


def test_agent_activity_and_generation_renderers_are_safe_and_actionable(
    tmp_path: Path,
) -> None:
    captured = _run_dashboard_node(
        tmp_path,
        r"""
        const attack = `"><img src=x onerror=globalThis.pwned=1>`;
        const numericAttack = `NaN${attack}`;
        const detail = {
          name: "AgentMarketCycle",
          recent_operations: [
            {seq: 12, role: `newest${attack}`, mode: attack, model: attack,
             generation: 2, cycle: 1, tokens: 30, latency_ms: 4.4, chars: 90},
            {seq: numericAttack, role: "older", mode: "full", model: "safe",
             generation: numericAttack, cycle: numericAttack, tokens: numericAttack,
             latency_ms: numericAttack, chars: numericAttack},
          ],
          generations: [
            {generation: 1, cycle: 0, generated: 7, passed: 6, qualified: 3,
             elite: 1, llm_calls: 4, wall_seconds: 3.2, elite_mean_score: -0.1},
            {generation: 2, cycle: 1, generated: numericAttack, passed: numericAttack,
             qualified: numericAttack, elite: numericAttack, llm_calls: numericAttack,
             wall_seconds: numericAttack, elite_mean_score: numericAttack},
          ],
        };
        const activity = renderAgentActivity(detail);
        const generations = renderAgentGenerations(detail);
        assert(activity.includes("最多显示最近 200 条"), "activity bound is not disclosed");
        for (const value of ["newest", "mode", "model", "Tokens", "字符"]) {
          assert(activity.includes(value), `activity omitted ${value}`);
        }
        assert(activity.indexOf("newest") < activity.indexOf("older"),
          "backend newest-first activity order was changed");
        assert(activity.includes('data-action="call" data-seq="12"'),
          "valid call does not expose a normalized action");
        assert((activity.match(/data-action="call"/g) || []).length === 1,
          "invalid call id became actionable");
        assert(generations.includes('data-action="generation"') &&
          generations.includes('data-agent="AgentMarketCycle" data-gen="2"'),
          "generation drill-down action is missing");
        assert(generations.indexOf('data-gen="2"') < generations.indexOf('data-gen="1"'),
          "generations were not rendered newest-first");
        for (const value of ["Generated", "Passed", "Qualified", "Elite", "LLM Calls", "耗时"]) {
          assert(generations.includes(value), `generation row omitted ${value}`);
        }
        const emptyActivity = renderAgentActivity({recent_operations: []});
        const emptyGenerations = renderAgentGenerations({generations: []});
        assert(emptyActivity.includes("该 Agent 暂无近期调用"), "activity empty state is not scoped");
        assert(emptyGenerations.includes("该 Agent 暂无代次记录"), "generation empty state is not scoped");

        const html = activity + generations;
        assert(html.includes("&lt;img") && !/<img\b/i.test(html), "archive text was not escaped");
        assert(!/NaN|Infinity|undefined/.test(html), "Agent rows leaked numeric garbage");
        assert(!globalThis.pwned, "malicious activity payload executed");
        console.log(JSON.stringify({html}));
        """,
    )
    generated = json.loads(captured)["html"]
    parser = _GeneratedMarkupParser()
    parser.feed(generated)

    assert not {"img", "script", "iframe", "object", "embed"}.intersection(parser.tags)
    assert not any(attribute.lower().startswith("on") for attribute in parser.attributes)


def test_agent_fetch_lifecycle_handles_error_retry_supersession_and_close(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const tabNames = ["overview", "activity", "generations"];
          const tabs = tabNames.map((name) => {
            const tab = new Element(`tab-${name}`);
            tab.dataset.tab = name;
            tab.parentElement = $("agent-tabs");
            return tab;
          });
          $("agent-tabs").childrenForQuery = tabs;
          tabNames.forEach((name) => {
            const panel = $(`agent-tabpanel-${name}`);
            panel.parentElement = $("agent-panel-body");
          });
          const card = new Element("agent-card");
          card.dataset.agent = "AgentMarketCycle";
          $("matrix").childrenForQuery = [card];
          LAST = {agents: [{
            name: "AgentMarketCycle", level: 1, selected: true, status: "running",
            generations: 0, llm_calls: 0, n_generated: 0, n_qualified: 0, n_elite: 0,
          }]};
          const pending = [];
          fetch = (url, options = {}) => new Promise((resolve) => {
            pending.push({url, options, resolve});
          });

          const failed = showAgent("AgentMarketCycle", card);
          assert(pending.length === 1 && pending[0].url === "/api/agent/AgentMarketCycle",
            "showAgent did not request its encoded endpoint");
          assert(pending[0].options.signal && DETAIL_ABORT === null,
            "Agent fetch reused the compatibility request controller");
          assert(AGENT_VIEW && AGENT_VIEW.name === "AgentMarketCycle" &&
            AGENT_VIEW.tab === "overview" && AGENT_VIEW.loading,
            "Agent panel lifecycle state was not initialized");
          assert($("agent-panel").classList.contains("open") &&
            $("agent-tabpanel-overview").innerHTML.includes("agent-skeleton"),
            "initial Agent load did not open with a skeleton");
          pending[0].resolve(response("archive unavailable", false));
          await failed;
          assert($("agent-tabpanel-overview").innerHTML.includes("Agent 详情加载失败") &&
            $("agent-tabpanel-overview").innerHTML.includes('data-action="retry-agent"'),
            "Agent fetch error lacks panel-local delegated retry");

          const detail = {
            name: "AgentMarketCycle", display_name: "Market Cycle", level: 1,
            selected: true, status: "running", current_generation: 1, current_cycle: 0,
            focus: "focus", probe: "probe",
            summary: {generations: 1, generated: 2, passed: 1, qualified: 1, elite: 0,
              llm_calls: 1, llm_tokens: 4, seconds: 2},
            trajectory: [], generations: [], recent_operations: [{seq: 4}],
          };
          const retry = new Element("retry");
          retry.dataset.action = "retry-agent";
          retry.dataset.agent = "AgentMarketCycle";
          retry.parentElement = $("agent-panel-body");
          const retried = activateArchiveAction(retry);
          assert(pending.length === 2, "delegated retry did not issue one request");
          pending[1].resolve(response(detail));
          await retried;
          assert(AGENT_VIEW.detail === detail && !AGENT_VIEW.loading,
            "successful retry did not commit detail");
          assert($("agent-panel-title").textContent === "Market Cycle" &&
            $("agent-panel-subtitle").innerHTML.includes("识别长期趋势"),
            "Agent header does not use endpoint identity and Chinese summary");

          const superseded = loadAgentDetail("AgentMarketCycle");
          const newest = loadAgentDetail("AgentMarketCycle");
          assert(pending[2].options.signal.aborted,
            "a superseded Agent request was not aborted");
          pending[3].resolve(response({...detail, display_name: "Newest"}));
          await newest;
          pending[2].resolve(response({...detail, display_name: "Stale"}));
          await superseded;
          assert($("agent-panel-title").textContent === "Newest",
            "stale Agent response overwrote the newest detail");

          const lateLoad = loadAgentDetail("AgentMarketCycle");
          const late = pending[4];
          closeAgentPanel();
          assert(late.options.signal.aborted, "closing did not cancel the Agent fetch");
          late.resolve(response({...detail, display_name: "Late"}));
          await lateLoad;
          assert(!AGENT_VIEW && $("agent-panel").hidden &&
            !$("agent-panel").classList.contains("open"),
            "stale completion reopened the closed Agent panel");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_agent_retry_restores_focus_without_stealing_it_on_live_refresh(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const tabNames = ["overview", "activity", "generations"];
          const tabs = tabNames.map((name) => {
            const tab = new Element(`tab-${name}`);
            tab.dataset.tab = name;
            tab.parentElement = $("agent-tabs");
            tab.setAttribute("aria-selected", String(name === "activity"));
            tab.tabIndex = name === "activity" ? 0 : -1;
            return tab;
          });
          $("agent-tabs").childrenForQuery = tabs;
          const body = $("agent-panel-body");
          const panels = tabNames.map((name) => $(`agent-tabpanel-${name}`));
          const retryByTab = {};
          panels.forEach((panel, index) => {
            const name = tabNames[index];
            panel.parentElement = body;
            let markup = "";
            Object.defineProperty(panel, "innerHTML", {
              configurable: true,
              get() { return markup; },
              set(value) {
                markup = value;
                if (value.includes('data-action="retry-agent"')) {
                  const retry = new Element(`retry-agent-${name}`);
                  retry.dataset.action = "retry-agent";
                  retry.dataset.agent = "AgentMarketCycle";
                  retry.parentElement = panel;
                  panel.childrenForQuery = [retry];
                  retryByTab[name] = retry;
                } else {
                  panel.childrenForQuery = [];
                }
                body.childrenForQuery = panels.flatMap((item) => item.childrenForQuery);
              },
            });
          });
          $("agent-panel").hidden = false;
          $("agent-panel").inert = false;
          $("agent-panel").classList.add("open");
          LAST = {agents: [{
            name: "AgentMarketCycle", level: 1, selected: true, status: "running",
            generations: 0, llm_calls: 0, n_generated: 0, n_qualified: 0, n_elite: 0,
          }]};
          AGENT_VIEW = {
            name: "AgentMarketCycle", tab: "activity", detail: null,
            loading: false, error: "首次失败", lastLoadedAt: 0,
            revision: agentRevision(LAST, "AgentMarketCycle"), refreshTimer: null,
          };
          renderAgentDetail();
          const firstRetry = retryByTab.activity;
          firstRetry.focus();

          const pending = [];
          fetch = (url, options = {}) => new Promise((resolve) => {
            pending.push({url, options, resolve});
          });
          const failedRetry = activateArchiveAction(firstRetry);
          pending[0].resolve(response("仍然失败", false));
          await failedRetry;
          assert(retryByTab.activity !== firstRetry &&
            document.activeElement === retryByTab.activity &&
            document.activeElement.parentElement === $("agent-tabpanel-activity") &&
            !$("agent-tabpanel-activity").hidden,
            "failed retry did not focus the retry in the visible Activity panel");

          const detail = {
            name: "AgentMarketCycle", display_name: "MarketCycle", level: 1,
            selected: true, status: "running", current_generation: 0, current_cycle: 0,
            focus: "focus", probe: "probe",
            summary: {generations: 0, generated: 0, passed: 0, qualified: 0,
              elite: 0, llm_calls: 0, llm_tokens: 0, seconds: 0},
            trajectory: [], generations: [], recent_operations: [],
          };
          const secondRetry = retryByTab.activity;
          const successfulRetry = activateArchiveAction(secondRetry);
          pending[1].resolve(response(detail));
          await successfulRetry;
          assert(document.activeElement === tabs[1] && AGENT_VIEW.tab === "activity",
            "successful retry did not focus the currently selected tab");

          const external = new Element("external-focus");
          external.focus();
          const liveRefresh = loadAgentDetail("AgentMarketCycle", {refresh: true});
          pending[2].resolve(response({...detail, current_generation: 1}));
          await liveRefresh;
          assert(document.activeElement === external,
            "ordinary live refresh stole focus from an unrelated control");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_agent_tabs_and_change_triggered_refresh_are_deterministic(
    tmp_path: Path,
) -> None:
    render_source = SCRIPT.split("function render(s)", 1)[1].split(
        "function renderMatrix", 1
    )[0]
    assert re.search(r"renderCands\(s\);\s*scheduleAgentRefresh\(\);", render_source)
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          let now = 1000;
          Date.now = () => now;
          const timers = new Map();
          let timerId = 0;
          setTimeout = (callback, delay) => {
            timers.set(++timerId, {callback, delay});
            return timerId;
          };
          clearTimeout = (id) => { timers.delete(id); };
          const tabNames = ["overview", "activity", "generations"];
          const tabs = tabNames.map((name) => {
            const tab = new Element(`tab-${name}`);
            tab.dataset.tab = name;
            tab.parentElement = $("agent-tabs");
            return tab;
          });
          $("agent-tabs").childrenForQuery = tabs;
          tabNames.forEach((name) => {
            $(`agent-tabpanel-${name}`).parentElement = $("agent-panel-body");
          });
          const card = new Element("agent-card");
          card.dataset.agent = "AgentMarketCycle";
          $("matrix").childrenForQuery = [card];
          const cardState = (generations, calls = 0) => ({
            name: "AgentMarketCycle", level: 1, selected: true, status: "running",
            generations, llm_calls: calls, n_generated: generations * 2,
            n_qualified: generations, n_elite: 0,
          });
          LAST = {agents: [cardState(0)]};
          const detail = {
            name: "AgentMarketCycle", display_name: "MarketCycle", level: 1,
            selected: true, status: "running", current_generation: 0, current_cycle: 0,
            focus: "focus", probe: "probe",
            summary: {generations: 1, generated: 2, passed: 1, qualified: 1, elite: 0,
              llm_calls: 1, llm_tokens: 5, seconds: 1},
            trajectory: [], generations: [], recent_operations: [{
              seq: 7, role: "judge", mode: "full", model: "mock",
              generation: 0, cycle: 0, tokens: 5, latency_ms: 1, chars: 3,
            }],
          };
          let fetchCount = 0;
          fetch = async () => { fetchCount += 1; return response(detail); };
          await showAgent("AgentMarketCycle", card);
          assert(fetchCount === 1 && timers.size === 0,
            "initial load armed an unchanged refresh");

          switchAgentTab("activity");
          assert(AGENT_VIEW.tab === "activity", "tab state was not retained");
          assert(tabs[1].attributes["aria-selected"] === "true" && tabs[1].tabIndex === 0,
            "selected tab lacks roving state");
          assert(tabs[0].attributes["aria-selected"] === "false" && tabs[0].tabIndex === -1,
            "inactive tab remains in the roving order");
          assert(!$("agent-tabpanel-activity").hidden &&
            $("agent-tabpanel-overview").hidden && $("agent-tabpanel-generations").hidden,
            "tab switch did not leave exactly one active panel");

          scheduleAgentRefresh();
          assert(timers.size === 0, "unchanged revision scheduled a refresh");
          LAST = {agents: [cardState(1)]};
          scheduleAgentRefresh();
          scheduleAgentRefresh();
          assert(timers.size === 1, "changed revision did not debounce to one timer");
          const firstTimer = Array.from(timers.entries())[0];
          assert(firstTimer[1].delay === AGENT_REFRESH_MIN_MS,
            "first refresh violated the two-second minimum interval");

          let resolveRefresh;
          fetch = () => {
            fetchCount += 1;
            return new Promise((resolve) => { resolveRefresh = resolve; });
          };
          now = 3000;
          timers.delete(firstTimer[0]);
          const refreshing = firstTimer[1].callback();
          assert(fetchCount === 2 && AGENT_VIEW.loading,
            "debounced change did not start exactly one refresh");
          assert(AGENT_VIEW.tab === "activity" &&
            !$("agent-tabpanel-activity").innerHTML.includes("agent-skeleton"),
            "live refresh stole the tab or replaced content with a skeleton");
          resolveRefresh(response({...detail, current_generation: 1}));
          await refreshing;
          assert(AGENT_VIEW.tab === "activity" && !AGENT_VIEW.loading,
            "completed refresh did not preserve the current tab");

          now = 3100;
          LAST = {agents: [cardState(2, 3)]};
          scheduleAgentRefresh();
          const secondTimer = Array.from(timers.entries())[0];
          assert(secondTimer[1].delay === 1900,
            "subsequent refresh was not delayed from lastLoadedAt");
          const cancelledCallback = secondTimer[1].callback;
          const beforeClose = fetchCount;
          closeAgentPanel();
          assert(timers.size === 0, "closing left the refresh timer armed");
          await cancelledCallback();
          assert(fetchCount === beforeClose, "cancelled refresh fetched after close");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_agent_refresh_does_not_mislabel_an_inflight_response_after_sse_change(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          let now = 1000;
          Date.now = () => now;
          const timers = new Map();
          let timerId = 0;
          setTimeout = (callback, delay) => {
            timers.set(++timerId, {callback, delay});
            return timerId;
          };
          clearTimeout = (id) => { timers.delete(id); };

          const tabNames = ["overview", "activity", "generations"];
          const tabs = tabNames.map((name) => {
            const tab = new Element(`tab-${name}`);
            tab.dataset.tab = name;
            tab.parentElement = $("agent-tabs");
            return tab;
          });
          $("agent-tabs").childrenForQuery = tabs;
          tabNames.forEach((name) => {
            $(`agent-tabpanel-${name}`).parentElement = $("agent-panel-body");
          });
          const card = new Element("agent-card");
          card.dataset.agent = "AgentMarketCycle";
          $("matrix").childrenForQuery = [card];
          const cardState = (generation) => ({
            name: "AgentMarketCycle", level: 1, selected: true, status: "running",
            generations: generation, llm_calls: generation,
            n_generated: generation * 2, n_qualified: generation, n_elite: 0,
          });
          const snapshot = (generation) => ({agents: [cardState(generation)]});
          const detail = (generation) => ({
            name: "AgentMarketCycle", display_name: "MarketCycle", level: 1,
            selected: true, status: "running", current_generation: generation,
            current_cycle: 0, focus: "focus", probe: "probe",
            summary: {generations: generation, generated: generation * 2, passed: generation,
              qualified: generation, elite: 0, llm_calls: generation,
              llm_tokens: generation * 4, seconds: generation},
            trajectory: [], generations: [], recent_operations: [],
          });

          LAST = snapshot(0);
          const pending = [];
          fetch = (url, options = {}) => new Promise((resolve) => {
            pending.push({url, options, resolve});
          });
          const initial = showAgent("AgentMarketCycle", card);
          switchAgentTab("activity", true);
          const activityTab = tabs[1];
          assert(document.activeElement === activityTab && AGENT_VIEW.tab === "activity",
            "test setup did not focus the Activity tab");

          LAST = snapshot(1);
          scheduleAgentRefresh();
          scheduleAgentRefresh();
          assert(timers.size === 0 && pending.length === 1,
            "SSE change started a duplicate request while R0 was in flight");

          pending[0].resolve(response(detail(0)));
          await initial;
          const revision0 = agentRevision(snapshot(0), "AgentMarketCycle");
          assert(AGENT_VIEW.revision === revision0,
            "R0 response was mislabeled with the mutable R1 snapshot revision");
          assert(timers.size === 1,
            "completion did not schedule the missed R1 refresh");
          const scheduled = Array.from(timers.entries())[0];
          assert(scheduled[1].delay === AGENT_REFRESH_MIN_MS,
            "catch-up refresh bypassed the two-second throttle");
          assert(AGENT_VIEW.tab === "activity" && document.activeElement === activityTab,
            "R0 completion changed the tab or stole focus");

          now = 3000;
          timers.delete(scheduled[0]);
          const catchUp = scheduled[1].callback();
          assert(pending.length === 2 && AGENT_VIEW.loading,
            "throttled catch-up did not issue exactly one R1 request");
          scheduleAgentRefresh();
          assert(pending.length === 2 && timers.size === 0,
            "R1 in-flight state created a duplicate request or timer");
          pending[1].resolve(response(detail(1)));
          await catchUp;

          assert(AGENT_VIEW.revision === agentRevision(LAST, "AgentMarketCycle"),
            "R1 response did not settle at the current revision");
          assert(timers.size === 0 && pending.length === 2,
            "settled R1 response armed an unnecessary refresh");
          assert(AGENT_VIEW.tab === "activity" && document.activeElement === activityTab,
            "catch-up refresh changed the tab or stole focus");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_agent_card_helpers_are_declared() -> None:
    for helper in ("stateLabel", "agentProgress", "bindAgentCards"):
        assert re.search(rf"function {helper}\b", HTML)


def test_backend_event_timestamp_is_converted_from_seconds_for_javascript() -> None:
    render_source = SCRIPT.split("function render", 1)[1].split("function renderMatrix", 1)[0]
    assert "finiteNumber(s.last_event_at)" in render_source
    assert re.search(r"new Date\([^)]*\* 1000\)", render_source)


def test_static_markup_has_unique_ids() -> None:
    ids = _parse_markup().ids
    duplicates = {element_id for element_id in ids if ids.count(element_id) > 1}

    assert ids
    assert duplicates == set()


def test_dashboard_has_no_external_asset_dependencies() -> None:
    markup = _parse_markup()

    assert markup.script_sources == []
    assert markup.link_hrefs == []
    assert re.search(r"url\(\s*['\"]?\s*https?://", STYLE, re.IGNORECASE) is None
    assert re.search(r"https?://", HTML, re.IGNORECASE) is None


def test_dashboard_includes_responsive_and_reduced_motion_rules() -> None:
    assert re.search(r"@media\s*\(max-width\s*:\s*1180px\)", STYLE)
    assert re.search(r"@media\s*\(max-width\s*:\s*760px\)", STYLE)
    assert re.search(r"@media\s*\(prefers-reduced-motion\s*:\s*reduce\)", STYLE)


def test_agent_panel_has_material_detail_layout_and_desktop_width() -> None:
    panel_rule = _css_rule("#agent-panel, #drawer")
    assert re.search(r"width\s*:\s*clamp\([^;]*60vw[^;]*\)", panel_rule)
    for selector in (
        ".agent-summary-grid",
        ".agent-summary-card",
        ".agent-activity-row",
        ".agent-generation-row",
        ".agent-skeleton",
        ".agent-error",
        ".agent-empty",
    ):
        assert _css_rule(selector).strip()
    mobile = re.search(
        r"@media\s*\(max-width\s*:\s*760px\)(.*?)"
        r"@media\s*\(prefers-reduced-motion",
        STYLE,
        re.DOTALL,
    )
    assert mobile is not None
    assert re.search(r"#agent-panel,\s*#drawer\s*\{[^}]*width\s*:\s*100vw", mobile.group(1))


def test_market_direction_colors_are_separate_from_status_semantics() -> None:
    tokens = dict(re.findall(r"--([a-z-]+):([^;]+);", STYLE))
    semantic_values = {
        tokens["md-primary"],
        tokens["md-success"],
        tokens["md-warning"],
        tokens["md-error"],
    }

    assert tokens["market-up"] != tokens["market-down"]
    assert tokens["market-up"] not in semantic_values
    assert tokens["market-down"] not in semantic_values
    assert _css_rule(".pos").strip() == "color:var(--market-up);"
    assert _css_rule(".neg").strip() == "color:var(--market-down);"

    status_rules = {
        ".status-chip.live": "--md-primary",
        ".status-chip.done": "--md-success",
        ".agent-card.running .agent-state-dot": "--md-primary",
        ".agent-card.done .agent-state-dot": "--md-success",
    }
    for selector, expected_token in status_rules.items():
        rule = _css_rule(selector)
        assert f"var({expected_token})" in rule
        assert "--market-" not in rule


def test_archive_origin_text_is_escaped_at_representative_render_boundaries() -> None:
    matrix = HTML.split("function renderMatrix", 1)[1].split("function renderFunnel", 1)[0]
    call_detail = HTML.split("async function showCall", 1)[1].split("function showAgent", 1)[0]
    alpha_detail = HTML.split("async function showAlpha", 1)[1].split("async function showCandidate", 1)[0]
    candidate_detail = HTML.split("async function showCandidate", 1)[1].split("// ------------------------------------------------------------------- transport", 1)[0]

    for pattern in ('data-agent="${esc(a.name)}"', "${esc(label)}"):
        assert pattern in matrix
    for pattern in (
        '${esc(d.system || "(none)")}',
        "${esc(d.prompt)}",
        "${esc(d.response)}",
    ):
        assert pattern in call_detail
    for pattern in (
        "openDrawer(esc(d.name)",
        "${esc(d.tier)}",
        "${esc(l.op)}",
        "${esc(l.agent)}",
        '${esc(l.guidance_mode || "?")}',
        '${esc(parents.join(", "))}',
        "${esc(d.rejected_at)}",
        "${esc(d.reject_reason)}",
        "${esc(c.stage)}",
        '${esc(c.detail || "")}',
        "${esc(d.code)}",
    ):
        assert pattern in alpha_detail
    assert "openDrawer(esc(safeFile), `<pre>${esc(d.code)}</pre>`)" in candidate_detail


def test_malicious_archive_payloads_render_as_inert_text(tmp_path: Path) -> None:
    captured = _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const attack = `"><img src=x onerror=globalThis.pwned=1>`;
          const apostrophe = `';globalThis.pwned=2;//`;
          const numericAttack = `7${attack}${apostrophe}`;
          const state = {
            finished: false, live: true, run_name: attack,
            config: {market: attack, generations: 4},
            totals: {
              llm_calls: numericAttack, llm_tokens: numericAttack,
              tokens_per_call: numericAttack, elapsed_seconds: numericAttack,
              unique_structures: numericAttack,
            },
            agents_done: numericAttack, agents_total: numericAttack,
            generations_seen: numericAttack, generations_planned: numericAttack,
            warnings: [attack], current_agent: attack,
            current_generation: numericAttack, current_cycle: numericAttack,
            agents: [{
              name: `Agent${attack}`, level: 1, selected: true, status: "running",
              generations: numericAttack, n_generated: numericAttack,
              n_passed: numericAttack, n_qualified: numericAttack,
              n_elite: numericAttack, llm_calls: numericAttack,
              seconds: numericAttack, best_score: numericAttack,
              best_rank_ic: numericAttack, stopped_early: attack,
            }],
            funnel: [{stage: attack, survivors: numericAttack, dropped: numericAttack}],
            latest_generation: {agent: attack, generation: numericAttack},
            elite_trajectory: [{
              agent: `Agent${attack}`, generation: numericAttack,
              cycle: numericAttack, n_elite: numericAttack, score: 0.1,
            }], plateau: {},
            calls_by_role: [{
              role: attack, calls: numericAttack, tokens: numericAttack,
              token_share: numericAttack, mean_latency_ms: numericAttack,
            }],
            recent_calls: [{
              seq: numericAttack, role: attack, agent: `Agent${attack}`,
              generation: numericAttack, tokens: numericAttack,
              latency_ms: numericAttack,
            }],
            candidates: [{
              file: attack, ic: numericAttack, rankic: numericAttack,
              icir: numericAttack, rankicir: numericAttack,
              mi: numericAttack, origin: attack,
            }],
          };
          render(state);
          showAgent(`Agent${attack}`);
          const shellHtml = [
            "cfg", "stats", "warnbox", "matrix", "funnel", "roles", "calls", "cands",
            "agent-panel-body",
          ].map((id) => $(id).innerHTML).join("\n");

          const detailBodies = {
            call: {
              tags: {role: attack, note: attack}, model: attack,
              temperature: attack, latency_ms: numericAttack,
              usage: {prompt_tokens: numericAttack, completion_tokens: numericAttack},
              system: attack, prompt: attack, response: attack,
            },
            generation: [{
              alpha_id: attack, name: attack, tier: attack, op: attack,
              fitness: {rank_ic: numericAttack}, rejected_at: attack,
              reject_reason: attack,
            }],
            alpha: {
              name: attack, tier: attack,
              fitness: {
                ic: numericAttack, rank_ic: numericAttack, icir: numericAttack,
                rank_icir: numericAttack, mi: numericAttack,
                n_days: numericAttack, nan_ratio: numericAttack,
              },
              lineage: {
                op: attack, agent: attack, level: numericAttack,
                guidance_mode: attack, generation: numericAttack,
                cycle: numericAttack, parents: [attack],
                repair_rounds: numericAttack, improve_rounds: numericAttack,
              },
              rejected_at: attack, reject_reason: attack,
              checks: [{passed: false, stage: attack, detail: attack}],
              code: attack,
            },
            candidate: {code: attack},
          };
          const urls = [];
          fetch = async (url) => {
            urls.push(url);
            if (url.startsWith("/api/call/")) return response(detailBodies.call);
            if (url.startsWith("/api/generation/")) return response(detailBodies.generation);
            if (url.startsWith("/api/alpha/")) return response(detailBodies.alpha);
            return response(detailBodies.candidate);
          };
          const detailHtml = [];
          await showCall(7); detailHtml.push($("dbody").innerHTML);
          await showGeneration(`Agent/${attack}`, 3); detailHtml.push($("dbody").innerHTML);
          await showAlpha(`alpha/${attack}`); detailHtml.push($("dbody").innerHTML);
          await showCandidate(`candidate/${attack}`); detailHtml.push($("dbody").innerHTML);

          const allHtml = shellHtml + detailHtml.join("\n");
          assert(!/<img\b/i.test(allHtml), "payload created an executable img tag");
          assert(!/<(?:img|svg|script|iframe|a)\b[^>]*\son\w+\s*=/i.test(allHtml),
            "payload created an inline event handler");
          assert(!/javascript\s*:/i.test(allHtml), "payload created a javascript URL");
          assert(allHtml.includes("&lt;img"), "payload was not preserved as escaped text");
          assert(urls.every((url) => !url.includes(attack)), "fetch URL contains an unencoded segment");
          assert(!globalThis.pwned, "malicious payload executed");
          console.log(JSON.stringify({html: allHtml}));
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )
    generated = json.loads(captured)["html"]
    parser = _GeneratedMarkupParser()
    parser.feed(generated)

    assert not {"img", "script", "iframe", "object", "embed"}.intersection(parser.tags)
    assert not any(attribute.lower().startswith("on") for attribute in parser.attributes)


def test_generated_archive_actions_are_keyboard_accessible(tmp_path: Path) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          renderCalls({recent_calls: [{seq: 4, role: "judge", agent: "AgentA", generation: 2}]});
          renderCands({candidates: [{file: "one.py", origin: "run"}]});
          fetch = async (url) => url.startsWith("/api/generation/")
            ? response([{alpha_id: "a1", name: "alpha", tier: "elite", fitness: {}}])
            : response({name: "alpha", fitness: {}, lineage: {}, checks: [], code: "x"});
          await showGeneration("AgentA", 2);
          const generationHtml = $("dbody").innerHTML;
          const generated = $("calls").innerHTML + $("cands").innerHTML + generationHtml;
          for (const action of ["call", "candidate", "alpha"]) {
            assert(generated.includes(`data-action="${action}"`), `missing ${action} action`);
          }
          assert((generated.match(/role="button"/g) || []).length >= 3, "actions lack button roles");
          assert((generated.match(/tabindex="0"/g) || []).length >= 3, "actions lack keyboard focus");
          assert(!/\son\w+\s*=/.test(generated), "generated rows use inline handlers");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_generation_without_saved_alphas_has_explicit_chinese_empty_state(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          fetch = async () => response([]);
          await showGeneration("AgentMarketCycle", 2);
          assert($("dbody").innerHTML.includes("该代暂无已保存的 Alpha 记录"),
            "missing alphas.jsonl is not explained in Chinese");
          assert(!$("dbody").innerHTML.includes("alphas.jsonl is only written"),
            "generation empty state still exposes implementation-only English text");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_dialogs_are_inert_when_closed_and_restore_focus_by_stable_agent(
    tmp_path: Path,
) -> None:
    assert re.search(
        r'<aside\b[^>]*id="agent-panel"[^>]*\bhidden\b[^>]*\binert\b', HTML
    )
    assert re.search(
        r'<aside\b[^>]*id="drawer"[^>]*role="dialog"[^>]*aria-modal="true"'
        r'[^>]*\bhidden\b[^>]*\binert\b',
        HTML,
    )
    _run_dashboard_node(
        tmp_path,
        r"""
        const oldCard = new Element("old-card");
        oldCard.dataset.agent = "AgentA";
        const newCard = new Element("new-card");
        newCard.dataset.agent = "AgentA";
        const matrix = $("matrix");
        matrix.childrenForQuery = [oldCard];
        $("agent-panel").hidden = true;
        $("agent-panel").inert = true;
        LAST = {agents: [{name: "AgentA", level: 1, selected: false}], elite_trajectory: []};
        showAgent("AgentA", oldCard);
        assert(!$("agent-panel").hidden && !$("agent-panel").inert, "open panel remains inert");
        matrix.childrenForQuery = [newCard];
        document.detached.add(oldCard);
        closeAgentPanel();
        assert($("agent-panel").hidden && $("agent-panel").inert, "closed panel remains tabbable");
        assert(document.activeElement === newCard, "focus was not restored to rerendered Agent card");

        const first = new Element("first");
        const last = new Element("last");
        const dialog = new Element("dialog");
        dialog.childrenForQuery = [first, last];
        document.activeElement = last;
        let prevented = false;
        trapDialogFocus({key: "Tab", shiftKey: false, preventDefault() { prevented = true; }}, dialog);
        assert(prevented && document.activeElement === first, "Tab did not wrap in dialog");
        """,
    )


def test_focus_trap_excludes_hidden_ancestor_and_recovers_external_focus(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        const dialog = new Element("dialog");
        const hiddenParent = new Element("hidden-parent");
        hiddenParent.hidden = true;
        hiddenParent.parentElement = dialog;
        const hiddenChild = new Element("hidden-child");
        hiddenChild.parentElement = hiddenParent;
        const invisible = new Element("invisible");
        invisible.parentElement = dialog;
        invisible.rects = [];
        const ariaHiddenParent = new Element("aria-hidden-parent");
        ariaHiddenParent.parentElement = dialog;
        ariaHiddenParent.setAttribute("aria-hidden", "true");
        const ariaHiddenChild = new Element("aria-hidden-child");
        ariaHiddenChild.parentElement = ariaHiddenParent;
        const cssHiddenParent = new Element("css-hidden-parent");
        cssHiddenParent.parentElement = dialog;
        cssHiddenParent.computedStyle.visibility = "hidden";
        const cssHiddenChild = new Element("css-hidden-child");
        cssHiddenChild.parentElement = cssHiddenParent;
        const disabled = new Element("disabled");
        disabled.parentElement = dialog;
        disabled.disabled = true;
        const negative = new Element("negative");
        negative.parentElement = dialog;
        negative.tabIndex = -1;
        const first = new Element("first");
        first.parentElement = dialog;
        const last = new Element("last");
        last.parentElement = dialog;
        dialog.childrenForQuery = [
          hiddenChild, invisible, ariaHiddenChild, cssHiddenChild,
          disabled, negative, first, last,
        ];

        document.activeElement = new Element("outside");
        let prevented = false;
        trapDialogFocus({key: "Tab", shiftKey: false, preventDefault() { prevented = true; }}, dialog);
        assert(prevented && document.activeElement === first,
          "external Tab did not recover to first truly tabbable control");

        document.activeElement = new Element("outside-again");
        prevented = false;
        trapDialogFocus({key: "Tab", shiftKey: true, preventDefault() { prevented = true; }}, dialog);
        assert(prevented && document.activeElement === last,
          "external Shift+Tab did not recover to last truly tabbable control");
        """,
    )


def test_nested_detail_suppresses_agent_modal_and_restores_generation_focus(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const agentCard = new Element("agent-card");
          agentCard.dataset.agent = "AgentA";
          $("matrix").childrenForQuery = [agentCard];
          LAST = {
            agents: [{name: "AgentA", level: 1, selected: true, status: "done"}],
            elite_trajectory: [{agent: "AgentA", generation: 2, cycle: 1, n_elite: 1, score: .1}],
          };
          showAgent("AgentA", agentCard);
          $("agent-panel").setAttribute("aria-modal", "true");

          const generation = new Element("generation");
          generation.dataset.action = "generation";
          generation.dataset.agent = "AgentA";
          generation.dataset.gen = "2";
          generation.parentElement = $("agent-panel-body");
          $("agent-panel-body").childrenForQuery = [generation];
          $("agent-panel").childrenForQuery = [$("agent-panel-body")];

          fetch = async (url) => url.startsWith("/api/generation/")
            ? response([{alpha_id: "alpha-1", name: "alpha", tier: "elite", fitness: {}}])
            : response({name: "alpha", fitness: {}, lineage: {}, checks: [], code: "x"});
          await activateArchiveAction(generation);
          assert($("agent-panel").inert, "underlying Agent modal remains interactive");
          assert($("agent-panel").attributes["aria-hidden"] === "true",
            "underlying Agent modal remains exposed");
          assert($("agent-panel").attributes["aria-modal"] !== "true",
            "two aria-modal dialogs are active");
          assert(!$("drawer").hidden && $("drawer").attributes["aria-hidden"] === "false",
            "detail drawer is not the active modal");

          const alpha = new Element("alpha");
          alpha.dataset.action = "alpha";
          alpha.dataset.alpha = "alpha-1";
          alpha.parentElement = $("dbody");
          $("dbody").childrenForQuery = [alpha];
          $("drawer").childrenForQuery = [$("dbody")];
          await activateArchiveAction(alpha);
          closeDrawer();

          assert(!$("agent-panel").inert && $("agent-panel").attributes["aria-hidden"] === "false",
            "Agent modal was not restored");
          assert($("agent-panel").attributes["aria-modal"] === "true",
            "Agent modal semantics were not restored");
          assert(document.activeElement === generation,
            "nested Alpha close did not return to stable generation control");
          assert(document.activeElement !== $("dclose"), "focus remained in hidden drawer");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_live_rerender_restores_only_focus_inside_changed_container(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        const matrix = $("matrix");
        const oldAgent = new Element("old-agent");
        oldAgent.dataset.agent = "AgentA";
        oldAgent.parentElement = matrix;
        const newAgent = new Element("new-agent");
        newAgent.dataset.agent = "AgentA";
        newAgent.parentElement = matrix;
        matrix.childrenForQuery = [oldAgent];
        let matrixHtml = "";
        Object.defineProperty(matrix, "innerHTML", {
          configurable: true,
          get() { return matrixHtml; },
          set(value) { matrixHtml = value; matrix.childrenForQuery = [newAgent]; },
        });
        document.activeElement = oldAgent;
        renderMatrix({agents: [{name: "AgentA", level: 1, selected: false}]});
        assert(document.activeElement === newAgent, "Agent focus was lost across rerender");

        const calls = $("calls");
        const oldCall = new Element("old-call");
        oldCall.dataset.action = "call";
        oldCall.dataset.seq = "9";
        oldCall.parentElement = calls;
        const newCall = new Element("new-call");
        newCall.dataset.action = "call";
        newCall.dataset.seq = "9";
        newCall.parentElement = calls;
        calls.childrenForQuery = [oldCall];
        let callsHtml = "";
        Object.defineProperty(calls, "innerHTML", {
          configurable: true,
          get() { return callsHtml; },
          set(value) { callsHtml = value; calls.childrenForQuery = [newCall]; },
        });
        document.activeElement = oldCall;
        renderCalls({recent_calls: [{seq: 9, role: "judge", agent: "AgentA"}]});
        assert(document.activeElement === newCall, "recent-call focus was lost across rerender");

        const external = new Element("external");
        document.activeElement = external;
        renderMatrix({agents: [{name: "AgentA", level: 1, selected: false}]});
        assert(document.activeElement === external, "rerender stole focus from another container");
        """,
    )


def test_agent_tabs_have_panels_keyboard_state_and_real_renderers() -> None:
    markup = _parse_markup()
    assert len(markup.data_tabs) == 3
    for tab in ("overview", "activity", "generations"):
        assert re.search(
            rf'<button\b[^>]*id="agent-tab-{tab}"[^>]*role="tab"'
            rf'[^>]*aria-controls="agent-tabpanel-{tab}"[^>]*tabindex="(?:0|-1)"',
            HTML,
        )
        assert re.search(
            rf'<[^>]+id="agent-tabpanel-{tab}"[^>]*role="tabpanel"', HTML
        )
    assert "ArrowLeft" in SCRIPT
    assert "ArrowRight" in SCRIPT
    for renderer in (
        "renderAgentOverview",
        "renderAgentActivity",
        "renderAgentGenerations",
    ):
        assert f"function {renderer}" in SCRIPT


def test_stale_or_closed_detail_requests_cannot_update_the_drawer(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const pending = [];
          fetch = (url, options = {}) => new Promise((resolve) => {
            pending.push({url, options, resolve});
          });
          $("drawer").hidden = true;
          $("drawer").inert = true;

          const first = showCall(1);
          const second = showCall(2);
          assert(pending.length === 2, "detail calls were not issued");
          assert(pending[0].options.signal, "detail fetch has no abort signal");
          assert(pending[0].options.signal.aborted, "new request did not abort the old request");
          pending[1].resolve(response({tags: {}, usage: {}, response: "second"}));
          await second;
          assert($("dbody").innerHTML.includes("second"), "newest response was not rendered");
          pending[0].resolve(response({tags: {}, usage: {}, response: "stale-first"}));
          await first;
          assert(!$("dbody").innerHTML.includes("stale-first"), "stale response overwrote newest detail");

          const third = showCandidate("late.py");
          const late = pending[2];
          closeDrawer();
          assert(late.options.signal.aborted, "closing did not abort the detail request");
          late.resolve(response({code: "late-response"}));
          await third;
          assert($("drawer").hidden && !$("drawer").classList.contains("open"), "late response reopened drawer");
          assert(!$("dbody").innerHTML.includes("late-response"), "late response updated closed drawer");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_sse_error_before_first_message_falls_back_once_and_ignores_stale_events(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const timers = new Map();
          let timerId = 0;
          setTimeout = (callback, delay) => { timers.set(++timerId, {callback, delay}); return timerId; };
          clearTimeout = (id) => { timers.delete(id); };
          const sources = [];
          EventSource = class {
            constructor() { this.closed = false; sources.push(this); }
            close() { this.closed = true; }
          };
          let fetchCount = 0;
          let resolveInitial;
          fetch = () => {
            fetchCount += 1;
            return new Promise((resolve) => { resolveInitial = resolve; });
          };
          connect();
          assert(sources.length === 1, "SSE was not constructed");
          sources[0].onerror();
          startPolling();
          assert(sources[0].closed, "pre-message SSE error did not close stale source");
          assert(POLLING, "pre-message SSE error did not start polling");
          assert(fetchCount === 1, "fallback/startPolling overlapped the initial state request");
          resolveInitial(response({run_name: "poll", agents: [], totals: {}}));
          await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
          const before = LAST && LAST.run_name;
          sources[0].onmessage({data: JSON.stringify({run_name: "stale-sse", agents: [], totals: {}})});
          assert(LAST && LAST.run_name === before, "stale SSE event rendered after polling fallback");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_sse_first_message_watchdog_and_post_success_error_behave_differently(
    tmp_path: Path,
) -> None:
    _run_dashboard_node(
        tmp_path,
        r"""
        (async () => {
          const timers = new Map();
          let timerId = 0;
          setTimeout = (callback, delay) => { timers.set(++timerId, {callback, delay}); return timerId; };
          clearTimeout = (id) => { timers.delete(id); };
          const sources = [];
          EventSource = class {
            constructor() { this.closed = false; sources.push(this); }
            close() { this.closed = true; }
          };
          let resolveInitial;
          fetch = () => new Promise((resolve) => { resolveInitial = resolve; });
          connect();
          assert(timers.size === 1, "first-message watchdog was not armed");
          const watchdog = Array.from(timers.values())[0];
          assert(watchdog.delay >= 3000 && watchdog.delay <= 5000,
            "first-message watchdog is not bounded to 3-5 seconds");
          watchdog.callback();
          assert(sources[0].closed && POLLING, "watchdog did not activate polling fallback");
          resolveInitial(response({run_name: "watchdog-poll", agents: [], totals: {}}));
          await Promise.resolve(); await Promise.resolve(); await Promise.resolve();

          POLLING = false; POLL_IN_FLIGHT = false; LAST = null;
          ACTIVE_EVENT_SOURCE = null; SSE_HAS_MESSAGE = false;
          timers.clear(); sources.length = 0;
          let resolveCold;
          fetch = () => new Promise((resolve) => { resolveCold = resolve; });
          connect();
          sources[0].onmessage({data: JSON.stringify({run_name: "sse", agents: [], totals: {}})});
          assert(timers.size === 0, "successful first message did not clear watchdog");
          sources[0].onerror();
          assert(!sources[0].closed, "post-success SSE error disabled native reconnect");
          assert(!POLLING, "post-success SSE error started duplicate polling");
          assert(/重连|中断/.test($("status").innerHTML), "post-success error lacks reconnecting status");
          resolveCold(response({run_name: "late-cold", agents: [], totals: {}}));
          await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
          assert(LAST.run_name === "sse", "late cold-state response overwrote SSE state");
        })().catch((error) => { console.error(error); process.exitCode = 1; });
        """,
    )


def test_detail_loading_uses_dedicated_live_region_and_hidden_scrim() -> None:
    detail_layer = re.search(r'<div\b[^>]*id="detail-layer"[^>]*>', HTML)
    assert detail_layer is not None
    assert "aria-live" not in detail_layer.group(0)
    assert re.search(
        r'<[^>]+id="detail-status"[^>]*aria-live="polite"[^>]*aria-atomic="true"',
        HTML,
    )
    scrim = re.search(r'<div\b[^>]*id="scrim"[^>]*>', HTML)
    assert scrim is not None
    assert 'aria-hidden="true"' in scrim.group(0)
    assert "hidden" in scrim.group(0)


def test_empty_funnel_clears_note_and_svg_uses_theme_tokens(tmp_path: Path) -> None:
    trajectory = SCRIPT.split("function renderTraj", 1)[1].split("function renderRoles", 1)[0]
    assert "themeColor(" in trajectory
    for hardcoded in ("#0b57d0", "#c7cdd5", "#e1e5eb", "#5f6368"):
        assert hardcoded not in trajectory
    _run_dashboard_node(
        tmp_path,
        r"""
        renderFunnel({
          funnel: [{stage: "generated", survivors: 2, dropped: 0}],
          latest_generation: {agent: "AgentA", generation: 1},
        });
        assert($("funnelnote").innerHTML, "setup did not render funnel note");
        renderFunnel({funnel: []});
        assert($("funnelnote").innerHTML === "", "empty funnel left a stale note");
        """,
    )


def test_all_api_path_segments_are_encoded_and_numeric_ids_normalized() -> None:
    assert re.search(r"function finiteInt\b", SCRIPT)
    assert re.search(
        r"fetch\(`/api/call/\$\{encodeURIComponent\([^}]+\)\}`", SCRIPT
    )
    assert re.search(
        r"fetch\(`/api/generation/\$\{encodeURIComponent\([^}]+\)\}/"
        r"\$\{encodeURIComponent\([^}]+\)\}`",
        SCRIPT,
    )
    assert re.search(
        r"fetch\(`/api/alpha/\$\{encodeURIComponent\([^}]+\)\}`", SCRIPT
    )
    assert re.search(
        r"fetch\(`/api/candidate/\$\{encodeURIComponent\([^}]+\)\}`", SCRIPT
    )
