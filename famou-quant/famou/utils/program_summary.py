"""Utilities for extracting compact program summaries for prompts and planning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Program


def get_improvement_directions(program: Optional[Program]) -> str:
    """Return normalized improvement directions from judge feedback."""
    if program is None or not isinstance(program.llm_feedback, dict):
        return ""

    feedback = program.llm_feedback
    value = feedback.get("improvement_directions") or feedback.get("improvements") or ""
    return str(value).strip()


def get_key_features(program: Optional[Program]) -> str:
    """Return normalized key features from judge feedback."""
    if program is None or not isinstance(program.llm_feedback, dict):
        return ""
    value = program.llm_feedback.get("key_features") or ""
    return str(value).strip()


def get_implementation_plan(program: Optional[Program]) -> str:
    """Return stored implementation plan text from program metadata."""
    if program is None:
        return ""
    value = program.meta.get("implementation_plan") if isinstance(program.meta, dict) else ""
    return str(value).strip() if value is not None else ""


def build_program_summary(
    program: Program,
    *,
    include_code: bool,
    code_char_limit: Optional[int] = None,
    relation: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a structured prompt-friendly summary for a program."""
    _ = code_char_limit  # Deprecated: code should never be truncated in summaries.
    code = program.code

    summary: Dict[str, Any] = {
        "program_id": program.id,
        "parent_id": program.parent_id,
        "generation": program.generation,
        "iteration": program.iteration,
        "combined_score": program.combined_score,
        "metrics": dict(program.metrics or {}),
        "validity": program.validity,
        "error_info": program.error_info,
        "key_features": get_key_features(program),
        "improvement_directions": get_improvement_directions(program),
        "implementation_plan": get_implementation_plan(program),
    }
    if relation:
        summary["relation"] = relation
    if include_code:
        summary["code"] = code
    return summary


def extract_eval_wall_time(program: Optional[Program]) -> Optional[float]:
    """Extract evaluation wall time from metrics, falling back to rollout timing."""
    if program is None:
        return None
    metrics = program.metrics if isinstance(program.metrics, dict) else {}
    metric_value = metrics.get("eval_wall_time")
    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
        return float(metric_value)
    post_time = program.post_time if isinstance(program.post_time, dict) else {}
    module_stats = post_time.get("module_stats")
    if not isinstance(module_stats, list):
        return None
    for module_stat in module_stats:
        if not isinstance(module_stat, dict):
            continue
        if module_stat.get("module_name") != "EvaluateModule":
            continue
        execution_time = module_stat.get("execution_time")
        if isinstance(execution_time, (int, float)) and not isinstance(execution_time, bool):
            return float(execution_time)
    return None


def build_anchor_summary(program: Program) -> Dict[str, Any]:
    """Build minimal anchor program summary for planner prompt payload."""
    return {
        "implementation_plan": get_implementation_plan(program),
        "combined_score": program.combined_score,
        "metrics": dict(program.metrics or {}),
        "eval_time": extract_eval_wall_time(program),
        "error_info": program.error_info,
    }


def normalize_plan_summary(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize rich plan metadata to minimal planner-facing schema."""
    if not isinstance(summary, dict):
        return None
    metrics = (
        summary.get("best_metrics")
        or summary.get("representative_metrics")
        or summary.get("metrics")
        or {}
    )
    if not isinstance(metrics, dict):
        metrics = {}
    combined_score = None
    for key in ("best_score", "representative_combined_score", "combined_score"):
        value = summary.get(key)
        if value is not None:
            combined_score = value
            break
    eval_time = None
    for key in ("best_eval_wall_time", "representative_eval_wall_time", "eval_time"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            eval_time = float(value)
            break
    if eval_time is None:
        metric_value = metrics.get("eval_wall_time")
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
            eval_time = float(metric_value)
    error_info = None
    for key in ("best_error_info", "representative_error_info", "error_info"):
        value = summary.get(key)
        if value:
            error_info = value
            break
    implementation_plan = ""
    for key in (
        "best_implementation_plan",
        "representative_implementation_plan",
        "implementation_plan",
    ):
        value = summary.get(key)
        if value:
            implementation_plan = str(value).strip()
            break
    return {
        "implementation_plan": implementation_plan,
        "combined_score": combined_score,
        "metrics": metrics,
        "eval_time": eval_time,
        "error_info": error_info,
    }


def infer_relation(reference: Program, other: Program) -> str:
    """Infer a simple genealogical relation label between two programs."""
    if other.id == reference.id:
        return "self"
    if other.parent_id == reference.id:
        return "child_of_parent"
    if reference.parent_id == other.id:
        return "parent_of_parent"
    if other.parent_id and reference.parent_id and other.parent_id == reference.parent_id:
        return "sibling_branch"
    if other.generation > reference.generation:
        return "later_plan_variant"
    if other.generation < reference.generation:
        return "earlier_plan_variant"
    return "peer_plan_variant"
