"""Prompt templates.

Every template opens with a ``[ROLE: xxx]`` marker.  That marker is load-bearing
in two places: the mock backend dispatches on it, and a reader of
``llm_calls.jsonl`` can filter a 100k-line transcript by stage without guessing.

Templates are plain module-level strings rather than files on disk so that the
package has no data-file dependency and a prompt can be diffed in git alongside
the code that formats it.
"""

from cogalpha.prompts.templates import (  # noqa: F401
    ALPHA_CONTRACT,
    ANALYSE_PROMPT,
    CODE_QUALITY_PROMPT,
    CROSSOVER_PROMPT,
    GENERATE_PROMPT,
    IMPROVE_PROMPT,
    JUDGE_PROMPT,
    MUTATE_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
)

__all__ = [
    "ALPHA_CONTRACT",
    "ANALYSE_PROMPT",
    "CODE_QUALITY_PROMPT",
    "CROSSOVER_PROMPT",
    "GENERATE_PROMPT",
    "IMPROVE_PROMPT",
    "JUDGE_PROMPT",
    "MUTATE_PROMPT",
    "REPAIR_PROMPT",
    "SYSTEM_PROMPT",
]
