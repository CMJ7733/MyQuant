"""
Strategy Router - auto-select strategy based on task direction.

Maps a high-level task direction to a concrete strategy name. The direction
can be explicitly set in config, or inferred from the task description by
an LLM call.

Directions:
    - machine_learning: ML/DL tasks (Kaggle, training, inference)
    - combinatorial_optimization: competitive programming, heuristic search
    - math: mathematical optimization, numerical computation
    - other: fallback bucket for tasks outside the three categories above

Usage in config.yaml:
    experiment:
      strategy_router:
        enabled: true
        direction: machine_learning  # optional, LLM infers if omitted
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_temperature,
    get_llm_timeout,
)

logger = logging.getLogger("famou")

# -------------------------------------------------------------------------
# Direction -> Strategy mapping
# -------------------------------------------------------------------------

DIRECTION_STRATEGY_MAP: Dict[str, str] = {
    "machine_learning": "example_strategy",
    "combinatorial_optimization": "adaptive_cluster",
    "math": "standard",
    "other": "example_strategy",
}

VALID_DIRECTIONS = list(DIRECTION_STRATEGY_MAP.keys())


@dataclass(frozen=True)
class StrategyRouteDecision:
    """Resolved router decision for the current experiment."""

    direction: str
    strategy_name: str
    source: str
    reasoning: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, str]:
        """Serialize decision for checkpoint persistence."""
        payload = {
            "direction": self.direction,
            "strategy_name": self.strategy_name,
            "source": self.source,
        }
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        return payload


# -------------------------------------------------------------------------
# LLM-based direction inference
# -------------------------------------------------------------------------

_DIRECTION_SYSTEM_PROMPT = """\
<system>
  <role>You are a strict task classifier.</role>

  <task>
    Classify the user's task into exactly one direction label.
  </task>

  <labels>
    <label name="machine_learning">
      Tasks centered on model training, prediction, feature engineering,
      neural networks, dataset-based learning, Kaggle-style pipelines,
      or inference systems.
    </label>
    <label name="combinatorial_optimization">
      Tasks centered on search, scheduling, packing, routing, graph algorithms,
      heuristic optimization, competitive programming, discrete decision-making,
      or NP-hard structure.
    </label>
    <label name="math">
      Tasks centered on mathematical derivation, symbolic reasoning,
      numerical analysis, formula manipulation, theorem/proof-style reasoning,
      or continuous mathematical optimization.
    </label>
    <label name="other">
      Tasks that do not clearly fit any of the three labels above.
      If uncertain, choose other instead of guessing.
    </label>
  </labels>

  <decision_rules>
    <rule>Choose the label based on the core task, not incidental wording.</rule>
    <rule>If the task is about training or evaluating models on data, prefer machine_learning.</rule>
    <rule>If the task is about discrete structure, search, graph, scheduling, packing, or heuristics, prefer combinatorial_optimization.</rule>
    <rule>If the task is mainly about formulas, derivations, proofs, or numerical mathematics, prefer math.</rule>
    <rule>If the task is ambiguous, mixed, or lacks a clear dominant category, choose other.</rule>
  </decision_rules>

  <output_format>
    Return exactly one XML document with this structure:
    <decision>
      <direction>machine_learning|combinatorial_optimization|math|other</direction>
      <reasoning>One short sentence explaining the choice.</reasoning>
    </decision>
  </output_format>

  <output_rules>
    <rule>Return exactly one decision element.</rule>
    <rule>direction must be one of: machine_learning, combinatorial_optimization, math, other.</rule>
    <rule>reasoning must be concise and plain text.</rule>
    <rule>Do not include markdown or code fences.</rule>
    <rule>Do not include any text before or after the decision element.</rule>
  </output_rules>
</system>"""

_DIRECTION_USER_PROMPT = """\
<input_task_description>
{task_description}
</input_task_description>"""


def _normalize_direction(raw: str) -> str:
    """Normalize LLM output / config input to direction key format."""
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def _llm_meta(llm_client) -> Dict[str, Any]:
    """Collect lightweight LLM client metadata for trace logging."""
    return {
        "client_class": type(llm_client).__name__ if llm_client is not None else None,
        "model": getattr(llm_client, "model", None) if llm_client is not None else None,
        "temperature": getattr(llm_client, "temperature", None) if llm_client is not None else None,
        "max_tokens": getattr(llm_client, "max_tokens", None) if llm_client is not None else None,
    }


def _resolve_direction_strategy_map(router_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Resolve the effective direction->strategy mapping with config overrides."""
    resolved = dict(DIRECTION_STRATEGY_MAP)
    if not isinstance(router_config, dict):
        return resolved

    override_map = router_config.get("direction2strategy", {})
    if not isinstance(override_map, dict):
        logger.warning(
            "[StrategyRouter] Ignoring non-dict strategy_router.direction2strategy=%r",
            override_map,
        )
        return resolved

    if not override_map:
        return resolved

    for raw_direction, raw_strategy in override_map.items():
        direction = _normalize_direction(str(raw_direction))
        strategy_name = str(raw_strategy).strip()
        if direction not in VALID_DIRECTIONS:
            logger.warning(
                "[StrategyRouter] Ignoring unknown direction2strategy key '%s'; valid directions: %s",
                raw_direction,
                VALID_DIRECTIONS,
            )
            continue
        if not strategy_name:
            logger.warning(
                "[StrategyRouter] Ignoring empty strategy override for direction '%s'",
                direction,
            )
            continue
        resolved[direction] = strategy_name
    return resolved


def _build_trace(
    *,
    experiment_config,
    direction_strategy_map: Optional[Dict[str, str]] = None,
    llm_client=None,
    llm_response=None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    raw_response: Optional[str] = None,
    parsed_direction: Optional[str] = None,
    parsed_reasoning: Optional[str] = None,
    final_direction: Optional[str] = None,
    final_strategy: Optional[str] = None,
    source: Optional[str] = None,
    fallback_applied: bool = False,
    fallback_reason: Optional[str] = None,
    parse_error: Optional[str] = None,
    persisted_decision: Optional[Dict[str, Any]] = None,
    attempts: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a structured strategy-router trace payload."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "task_description": experiment_config.task_description,
            "configured_direction": experiment_config.strategy_router.get("direction", ""),
            "configured_fallback_strategy": experiment_config.strategy,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "persisted_decision": persisted_decision or {},
            "direction_strategy_map": direction_strategy_map or DIRECTION_STRATEGY_MAP,
        },
        "output": {
            "raw_response": raw_response,
            "parsed_direction": parsed_direction,
            "parsed_reasoning": parsed_reasoning,
            "parse_error": parse_error,
            "attempts": attempts or [],
            "llm_response": {
                "model": getattr(llm_response, "model", None) if llm_response is not None else None,
                "finish_reason": getattr(llm_response, "finish_reason", None) if llm_response is not None else None,
                "usage": getattr(llm_response, "usage", None) if llm_response is not None else None,
                "provider_raw_response": (
                    getattr(llm_response, "raw_response", None)
                    if llm_response is not None
                    else None
                ),
            },
        },
        "decision": {
            "final_direction": final_direction,
            "final_strategy": final_strategy,
            "source": source,
            "fallback_applied": fallback_applied,
            "fallback_reason": fallback_reason,
        },
        "meta": _llm_meta(llm_client),
    }


def _parse_xml_decision(raw_response: str) -> Dict[str, Optional[str]]:
    """
    Parse a strict XML router decision.

    Expected shape:
    <decision>
      <direction>...</direction>
      <reasoning>...</reasoning>
    </decision>
    """
    root = ET.fromstring(raw_response)
    if root.tag != "decision":
        raise ValueError(f"expected root <decision>, got <{root.tag}>")

    child_tags = [child.tag for child in root if isinstance(child.tag, str)]
    expected_tags = {"direction", "reasoning"}
    unexpected_tags = sorted(set(child_tags) - expected_tags)
    if unexpected_tags:
        raise ValueError(f"unexpected decision fields: {unexpected_tags}")
    for tag in expected_tags:
        if child_tags.count(tag) != 1:
            raise ValueError(f"expected exactly one <{tag}> element")

    direction = _normalize_direction(root.findtext("direction", default=""))
    reasoning = (root.findtext("reasoning", default="") or "").strip() or None

    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"invalid direction '{direction}'")

    if not reasoning:
        raise ValueError("missing reasoning")

    return {
        "direction": direction,
        "reasoning": reasoning,
    }


def _decision_from_persisted(
    data: Dict[str, Any],
    direction_strategy_map: Dict[str, str],
    source: str = "checkpoint",
) -> Optional[StrategyRouteDecision]:
    """Reconstruct a persisted routing decision if it is valid."""
    if not data:
        return None

    direction = _normalize_direction(str(data.get("direction", "")))
    strategy_name = str(data.get("strategy_name", "")).strip()
    persisted_source = str(data.get("source", source)).strip() or source

    if direction not in VALID_DIRECTIONS:
        logger.warning(
            "[StrategyRouter] Ignoring persisted routing decision with invalid direction '%s'",
            direction,
        )
        return None

    expected_strategy = direction_strategy_map[direction]
    if strategy_name != expected_strategy:
        logger.warning(
            "[StrategyRouter] Ignoring persisted routing decision with mismatched strategy "
            "'%s' for direction '%s' (expected '%s')",
            strategy_name,
            direction,
            expected_strategy,
        )
        return None

    return StrategyRouteDecision(
        direction=direction,
        strategy_name=strategy_name,
        source=persisted_source,
        reasoning=str(data.get("reasoning", "")).strip() or None,
    )


def infer_direction(task_description: str, llm_client) -> str:
    """
    Use LLM to infer the task direction from the task description.

    Args:
        task_description: The task description from config
        llm_client: LLM client instance (with .generate() method)

    Returns:
        One of: machine_learning, combinatorial_optimization, math, other
    """
    class _TraceConfig:
        def __init__(self, task_description: str):
            self.task_description = task_description
            self.strategy_router = {}
            self.strategy = DIRECTION_STRATEGY_MAP["other"]

    return infer_direction_decision(_TraceConfig(task_description), llm_client).direction


def classify_task_direction(task_description: str, llm_client) -> str:
    """
    Public classification entry point: task description -> direction label.

    This is a lightweight wrapper for callers that only want the category label
    and do not want to construct an ExperimentConfig-like object.

    Args:
        task_description: Raw user task description
        llm_client: LLM client instance

    Returns:
        One of: machine_learning, combinatorial_optimization, math, other
    """
    return infer_direction(task_description, llm_client)


def infer_direction_decision(experiment_config, llm_client) -> StrategyRouteDecision:
    """
    Use LLM to infer the task direction and return the full routing decision.

    Falls back to the `other` bucket if the model returns an unsupported label.
    """
    prompt = _DIRECTION_USER_PROMPT.format(
        task_description=experiment_config.task_description
    )
    direction_strategy_map = _resolve_direction_strategy_map(
        getattr(experiment_config, "strategy_router", {})
    )
    attempts: list[Dict[str, Any]] = []
    last_response = None
    last_raw_response = None
    last_error = None
    max_attempts = get_llm_max_retries(llm_client)
    temperature = get_llm_temperature(llm_client)
    max_tokens = get_llm_max_tokens(llm_client)
    timeout = get_llm_timeout(llm_client)

    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt = (
                f"{prompt}\n\n"
                "<retry_instruction>"
                "Your previous response could not be used. "
                "Return exactly one valid <decision> XML document and nothing else."
                "</retry_instruction>"
            )

        response = None
        try:
            response = llm_client.generate(
                prompt=attempt_prompt,
                system=_DIRECTION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_response = response.text.strip()
            parsed = _parse_xml_decision(raw_response)
            attempts.append(
                {
                    "attempt": attempt,
                    "success": True,
                    "raw_response": raw_response,
                    "parsed_direction": parsed["direction"],
                    "parsed_reasoning": parsed["reasoning"],
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_tokens": max_tokens,
                    "llm_response": {
                        "model": getattr(response, "model", None),
                        "finish_reason": getattr(response, "finish_reason", None),
                        "usage": getattr(response, "usage", None),
                        "provider_raw_response": getattr(response, "raw_response", None),
                    },
                }
            )
            logger.info(
                "[StrategyRouter] LLM inferred direction=%s -> strategy=%s",
                parsed["direction"],
                direction_strategy_map[parsed["direction"]],
            )
            return StrategyRouteDecision(
                direction=parsed["direction"] or "other",
                strategy_name=(
                    direction_strategy_map[parsed["direction"]]
                    if parsed["direction"]
                    else direction_strategy_map["other"]
                ),
                source="llm",
                reasoning=parsed["reasoning"],
                trace=_build_trace(
                    experiment_config=experiment_config,
                    direction_strategy_map=direction_strategy_map,
                    llm_client=llm_client,
                    llm_response=response,
                    system_prompt=_DIRECTION_SYSTEM_PROMPT,
                    user_prompt=attempt_prompt,
                    raw_response=raw_response,
                    parsed_direction=parsed["direction"],
                    parsed_reasoning=parsed["reasoning"],
                    final_direction=parsed["direction"],
                    final_strategy=direction_strategy_map[parsed["direction"]],
                    source="llm",
                    attempts=attempts,
                ),
            )
        except Exception as exc:
            last_response = response
            last_raw_response = (
                response.text.strip() if response is not None and getattr(response, "text", None) is not None else None
            )
            last_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "success": False,
                    "raw_response": last_raw_response,
                    "parse_error": str(exc),
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_tokens": max_tokens,
                    "llm_response": {
                        "model": getattr(last_response, "model", None) if last_response is not None else None,
                        "finish_reason": getattr(last_response, "finish_reason", None) if last_response is not None else None,
                        "usage": getattr(last_response, "usage", None) if last_response is not None else None,
                        "provider_raw_response": getattr(last_response, "raw_response", None) if last_response is not None else None,
                    },
                }
            )
            logger.warning(
                "[StrategyRouter] Attempt %s/%s failed to get a valid router response: %s",
                attempt,
                max_attempts,
                exc,
            )

    fallback_direction = "other"
    fallback_strategy = direction_strategy_map[fallback_direction]
    return StrategyRouteDecision(
        direction=fallback_direction,
        strategy_name=fallback_strategy,
        source="llm_fallback",
        reasoning="Fallback applied because the LLM response could not be parsed safely.",
        trace=_build_trace(
            experiment_config=experiment_config,
            direction_strategy_map=direction_strategy_map,
            llm_client=llm_client,
            llm_response=last_response,
            system_prompt=_DIRECTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            raw_response=last_raw_response,
            final_direction=fallback_direction,
            final_strategy=fallback_strategy,
            source="llm_fallback",
            fallback_applied=True,
            fallback_reason=last_error,
            parse_error=last_error,
            attempts=attempts,
        ),
    )


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------


def resolve_strategy_route(
    experiment_config,
    llm_client=None,
    persisted_decision: Optional[Dict[str, Any]] = None,
) -> StrategyRouteDecision:
    """
    Resolve the routing decision based on strategy_router config.

    Args:
        experiment_config: ExperimentConfig instance
        llm_client: LLM client (required if direction is not set)
        persisted_decision: Previously saved routing decision to reuse on resume

    Returns:
        Full routing decision with direction, strategy_name, and source.
    """
    router_config: Dict[str, Any] = experiment_config.strategy_router
    direction_strategy_map = _resolve_direction_strategy_map(router_config)
    direction = _normalize_direction(router_config.get("direction", ""))

    if direction and direction in VALID_DIRECTIONS:
        strategy_name = direction_strategy_map[direction]
        logger.info(
            "[StrategyRouter] Using configured direction=%s -> strategy=%s",
            direction,
            strategy_name,
        )
        return StrategyRouteDecision(
            direction=direction,
            strategy_name=strategy_name,
            source="config",
            trace=_build_trace(
                experiment_config=experiment_config,
                direction_strategy_map=direction_strategy_map,
                final_direction=direction,
                final_strategy=strategy_name,
                source="config",
            ),
        )
    elif direction and direction not in VALID_DIRECTIONS:
        logger.warning(
            f"[StrategyRouter] Invalid direction '{direction}', "
            f"valid options: {VALID_DIRECTIONS}. Falling back to LLM inference."
        )
        direction = ""

    checkpoint_decision = _decision_from_persisted(
        persisted_decision or {},
        direction_strategy_map=direction_strategy_map,
    )
    if checkpoint_decision is not None:
        logger.info(
            "[StrategyRouter] Reusing persisted direction=%s -> strategy=%s",
            checkpoint_decision.direction,
            checkpoint_decision.strategy_name,
        )
        return StrategyRouteDecision(
            direction=checkpoint_decision.direction,
            strategy_name=checkpoint_decision.strategy_name,
            source=checkpoint_decision.source,
            reasoning=checkpoint_decision.reasoning,
            trace=_build_trace(
                experiment_config=experiment_config,
                direction_strategy_map=direction_strategy_map,
                final_direction=checkpoint_decision.direction,
                final_strategy=checkpoint_decision.strategy_name,
                source=checkpoint_decision.source,
                persisted_decision=persisted_decision,
            ),
        )

    if not direction:
        if llm_client is None:
            raise ValueError(
                "strategy_router.direction is not set and no LLM client provided. "
                "Either set direction explicitly or ensure LLM is configured."
            )
        logger.info("[StrategyRouter] No direction configured, using LLM to infer...")
        return infer_direction_decision(experiment_config, llm_client)

    strategy_name = direction_strategy_map[direction]
    return StrategyRouteDecision(
        direction=direction,
        strategy_name=strategy_name,
        source="config",
        trace=_build_trace(
            experiment_config=experiment_config,
            direction_strategy_map=direction_strategy_map,
            final_direction=direction,
            final_strategy=strategy_name,
            source="config",
        ),
    )


def route_strategy(experiment_config, llm_client=None) -> str:
    """Backward-compatible helper returning only the selected strategy name."""
    return resolve_strategy_route(
        experiment_config,
        llm_client=llm_client,
    ).strategy_name
