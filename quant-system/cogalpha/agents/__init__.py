"""The Seven-Level Agent Hierarchy and its generation front-end (§3.1-§3.2)."""

from cogalpha.agents.hierarchy import (  # noqa: F401
    HIERARCHY,
    LAYER_DESCRIPTIONS,
    LAYERS,
    AgentSpec,
    by_level,
    get_agent,
    hierarchy_summary,
    select_agents,
)
from cogalpha.agents.guidance import (  # noqa: F401
    MODES,
    GuidanceMode,
    all_variants,
    build_guidance,
    get_mode,
    pick_mode,
)
from cogalpha.agents.generator import AlphaGenerator  # noqa: F401
from cogalpha.agents.parse import (  # noqa: F401
    ParseError,
    extract_blocks,
    function_name,
    parse_alphas,
    parse_verdict,
    rename_function,
)

__all__ = [
    "HIERARCHY",
    "LAYERS",
    "LAYER_DESCRIPTIONS",
    "AgentSpec",
    "by_level",
    "get_agent",
    "hierarchy_summary",
    "select_agents",
    "MODES",
    "GuidanceMode",
    "all_variants",
    "build_guidance",
    "get_mode",
    "pick_mode",
    "AlphaGenerator",
    "ParseError",
    "extract_blocks",
    "function_name",
    "parse_alphas",
    "parse_verdict",
    "rename_function",
]
