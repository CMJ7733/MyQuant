"""
Block detection utilities for Planning-based evolution.

Detects BLOCK markers in code:
- Python: # BLOCK A, # BLOCK B, etc.
- Other languages (C++/Java/JS/Go/Rust): // BLOCK A, // BLOCK B, etc.

Supports two formats:
1. Simple format: `// BLOCK A` ... `// BLOCK B` (next marker ends previous block)
2. START/END format: `// BLOCK A START: ...` ... `// BLOCK A END`

Features:
- Automatic language detection from Context
- Automatic format detection (simple vs START/END)
- Custom regex patterns
- Block extraction with span information
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Language to comment pattern mapping (simplified: Python vs others)
LANGUAGE_PATTERNS = {
    "python": r"#\s*BLOCK\s+([A-Z]+)",
    "default": r"//\s*BLOCK\s+([A-Z]+)",  # Java/C++/JavaScript/Go/Rust etc.
}

# Pattern for START/END format detection
START_END_PATTERNS = {
    "python": r"#\s*BLOCK\s+([A-Z]+)\s+START",
    "default": r"//\s*BLOCK\s+([A-Z]+)\s+START",
}


def get_block_pattern(language: str) -> str:
    """
    Get regex pattern for block detection based on language.

    Args:
        language: Programming language (e.g., "python", "cpp", "java")

    Returns:
        Regex pattern for detecting BLOCK markers
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        return LANGUAGE_PATTERNS["python"]
    return LANGUAGE_PATTERNS["default"]


def uses_start_end_format(code: str, language: str = "python") -> bool:
    """
    Check if code uses START/END format for block markers.

    Args:
        code: Source code to check
        language: Programming language

    Returns:
        True if code uses START/END format, False for simple format
    """
    if not code:
        return False

    lang = language.lower().strip()
    if lang in ("python", "py"):
        pattern = START_END_PATTERNS["python"]
    else:
        pattern = START_END_PATTERNS["default"]

    return bool(re.search(pattern, code, re.MULTILINE))


def detect_blocks_from_code(
    code: str,
    language: str = "python",
    pattern: Optional[str] = None,
    default_blocks: Optional[List[str]] = None,
) -> List[str]:
    """
    Detect block labels from code.

    Args:
        code: Source code with BLOCK markers
        language: Programming language for comment detection
        pattern: Custom regex pattern (overrides language-based pattern)
        default_blocks: Default blocks if none detected

    Returns:
        List of detected block labels (e.g., ["A", "B", "C"])

    Example:
        >>> code = '''
        ... # BLOCK A
        ... def func_a():
        ...     pass
        ...
        ... # BLOCK B
        ... def func_b():
        ...     pass
        ... '''
        >>> detect_blocks_from_code(code, language="python")
        ['A', 'B']
    """
    if not code:
        return list(default_blocks) if default_blocks else []

    # --- EVOLVE-BLOCK detection (before standard BLOCK detection) ---
    lang = language.lower().strip()
    if lang in ("python", "py"):
        evolve_comment = "#"
    else:
        evolve_comment = "//"

    evolve_start_pat = rf"{re.escape(evolve_comment)}\s*EVOLVE-BLOCK-START"
    evolve_end_pat = rf"{re.escape(evolve_comment)}\s*EVOLVE-BLOCK-END"

    starts = re.findall(evolve_start_pat, code, re.MULTILINE)
    ends = re.findall(evolve_end_pat, code, re.MULTILINE)

    if starts or ends:
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                f"Expected exactly one EVOLVE-BLOCK-START/END pair, "
                f"found {len(starts)} START and {len(ends)} END"
            )

        start_pos = re.search(evolve_start_pat, code).start()
        end_pos = re.search(evolve_end_pat, code).start()
        if start_pos >= end_pos:
            raise ValueError(
                "EVOLVE-BLOCK-START must appear before EVOLVE-BLOCK-END"
            )

        return ["EVOLVE"]

    # Use custom pattern or language-based pattern
    regex = pattern or get_block_pattern(language)

    # Find all block labels
    matches = re.findall(regex, code, re.MULTILINE)

    # Deduplicate while preserving order
    seen = set()
    blocks = []
    for label in matches:
        label = label.upper().strip()
        if label and label not in seen:
            seen.add(label)
            blocks.append(label)

    return blocks if blocks else (list(default_blocks) if default_blocks else [])


def parse_blocks_with_spans(
    code: str,
    language: str = "python",
    pattern: Optional[str] = None,
    force_start_end: Optional[bool] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse blocks with their span information (start, end positions).

    Used for block replacement operations.

    Supports two formats:
    1. Simple format: `// BLOCK A` ... `// BLOCK B` (next marker ends previous)
    2. START/END format: `// BLOCK A START: ...` ... `// BLOCK A END`

    The format is auto-detected unless force_start_end is specified.

    Args:
        code: Source code with BLOCK markers
        language: Programming language for comment detection
        pattern: Custom regex pattern (overrides language-based pattern)
        force_start_end: If True, force START/END format; if False, force simple format;
                         if None, auto-detect

    Returns:
        Dict mapping block labels to their info:
        {
            "A": {"span": (start, end), "text": "// BLOCK A\\n...", "format": "start_end"|"simple"},
            "B": {"span": (start, end), "text": "// BLOCK B\\n...", "format": "start_end"|"simple"},
        }

    Example:
        >>> blocks = parse_blocks_with_spans(code, language="python")
        >>> blocks["A"]["span"]  # (0, 50) - start and end positions
    """
    if not code:
        return {}

    # Determine comment prefix based on language
    lang = language.lower().strip()
    if lang in ("python", "py"):
        comment = "#"
    else:
        comment = "//"

    # Auto-detect format if not forced
    if force_start_end is None:
        use_start_end = uses_start_end_format(code, language)
    else:
        use_start_end = force_start_end

    blocks: Dict[str, Dict[str, Any]] = {}

    if use_start_end:
        # Parse START/END format: // BLOCK A START: ... // BLOCK A END
        # Pattern to find START markers
        start_pattern = rf"^[ \t]*{re.escape(comment)}\s*BLOCK\s+([A-Z]+)\s+START.*$"
        # Pattern to find END markers (with backreference)
        end_pattern_template = rf"^[ \t]*{re.escape(comment)}\s*BLOCK\s+{{label}}\s+END.*$"
        # Pattern for separator lines (e.g., //============)
        separator_pattern = rf"^[ \t]*{re.escape(comment)}=+[ \t]*$"

        for start_match in re.finditer(start_pattern, code, re.MULTILINE):
            label = start_match.group(1).upper().strip()
            start_pos = start_match.start()

            # Look backwards for separator lines before the START marker
            # Find start of line containing the START marker
            line_start = code.rfind('\n', 0, start_pos) + 1
            # Check previous lines for separators
            search_start = line_start
            while search_start > 0:
                prev_line_end = search_start - 1
                prev_line_start = code.rfind('\n', 0, prev_line_end) + 1
                prev_line = code[prev_line_start:prev_line_end]
                if re.match(separator_pattern, prev_line):
                    search_start = prev_line_start
                else:
                    break
            start_pos = search_start

            # Find corresponding END marker
            end_pattern = end_pattern_template.format(label=label)
            # Search from start_match.end() to avoid matching the start marker
            end_match = re.search(end_pattern, code[start_match.end():], re.MULTILINE)

            if end_match:
                # Calculate absolute end position (include the END marker line)
                end_pos = start_match.end() + end_match.end()
            else:
                # No END marker found, extend to end of code
                end_pos = len(code)

            blocks[label] = {
                "span": (start_pos, end_pos),
                "text": code[start_pos:end_pos],
                "format": "start_end",
            }
    else:
        # Parse simple format: // BLOCK A ... // BLOCK B
        if pattern:
            full_pattern = rf"^[ \t]*{re.escape(comment)}\s*BLOCK\s+([A-Z]+).*$"
        else:
            full_pattern = rf"^[ \t]*{re.escape(comment)}\s*BLOCK\s+([A-Z]+).*$"

        marker_positions: List[Tuple[int, str]] = []

        for match in re.finditer(full_pattern, code, re.MULTILINE):
            label = match.group(1).upper().strip()
            start = match.start()
            marker_positions.append((start, label))

        # Calculate spans (from this marker to next marker or end)
        for i, (start, label) in enumerate(marker_positions):
            if i + 1 < len(marker_positions):
                end = marker_positions[i + 1][0]
            else:
                end = len(code)

            blocks[label] = {
                "span": (start, end),
                "text": code[start:end],
                "format": "simple",
            }

    return blocks


def parse_child_blocks(
    response: str,
    language: str = "python",
) -> Dict[str, str]:
    """
    Parse BLOCK X START ... BLOCK X END markers from LLM response.

    Used to extract modified blocks from LLM output.

    Args:
        response: LLM response containing block modifications
        language: Programming language for comment detection

    Returns:
        Dict mapping block labels to their content:
        {
            "A": "def new_func_a():\\n    ...",
            "B": "def new_func_b():\\n    ...",
        }

    Example:
        >>> response = '''
        ... # BLOCK A START
        ... def new_func_a():
        ...     return 42
        ... # BLOCK A END
        ... '''
        >>> parse_child_blocks(response, language="python")
        {'A': 'def new_func_a():\\n    return 42'}
    """
    if not response:
        return {}

    # Determine comment prefix based on language
    lang = language.lower().strip()
    if lang in ("python", "py"):
        comment = "#"
    else:
        comment = "//"

    # Pattern for BLOCK X START ... BLOCK X END
    # Allows optional whitespace and captures content between markers
    pattern = rf"""
        ^[ \t]*{re.escape(comment)}\s*BLOCK\s+([A-Z]+)\s+START.*$
        ([\s\S]*?)
        ^[ \t]*{re.escape(comment)}\s*BLOCK\s+\1\s+END.*$
    """

    blocks: Dict[str, str] = {}
    for match in re.finditer(pattern, response, re.MULTILINE | re.VERBOSE):
        label = match.group(1).upper().strip()
        content = match.group(2).strip()
        blocks[label] = content

    if not blocks:
        evolve_pattern = rf"""
            ^[ \t]*{re.escape(comment)}\s*EVOLVE-BLOCK-START.*$
            ([\s\S]*?)
            ^[ \t]*{re.escape(comment)}\s*EVOLVE-BLOCK-END.*$
        """
        m = re.search(evolve_pattern, response, re.MULTILINE | re.VERBOSE)
        if m:
            blocks["EVOLVE"] = m.group(1).strip()

    return blocks


def replace_blocks(
    parent_code: str,
    parent_blocks: Dict[str, Dict[str, Any]],
    child_blocks: Dict[str, str],
    language: str = "python",
    block_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Replace blocks in parent code with child blocks.

    Performs replacement in reverse order (by span position) to avoid
    offset issues when multiple blocks are replaced.

    Preserves the original format (START/END or simple) from parent_blocks.

    Args:
        parent_code: Original code with blocks
        parent_blocks: Parent block info from parse_blocks_with_spans()
        child_blocks: Child block content from parse_child_blocks()
        language: Programming language for comment formatting
        block_descriptions: Optional descriptions for each block (used in START/END format)
                           e.g., {"A": "DESTROY OPERATORS", "B": "REPAIR OPERATORS"}

    Returns:
        Code with replaced blocks

    Example:
        >>> parent_blocks = parse_blocks_with_spans(parent_code, "python")
        >>> child_blocks = parse_child_blocks(llm_response, "python")
        >>> new_code = replace_blocks(parent_code, parent_blocks, child_blocks, "python")
    """
    if not child_blocks:
        return parent_code

    # Determine comment prefix
    lang = language.lower().strip()
    if lang in ("python", "py"):
        comment = "#"
    else:
        comment = "//"

    result = parent_code
    block_descriptions = block_descriptions or {}

    # Sort by span start position (descending) to replace from end to start
    sorted_items = sorted(
        [(label, content) for label, content in child_blocks.items()
         if label in parent_blocks],
        key=lambda x: parent_blocks[x[0]]["span"][0],
        reverse=True,
    )

    for label, content in sorted_items:
        start, end = parent_blocks[label]["span"]
        block_info = parent_blocks[label]
        block_format = block_info.get("format", "simple")

        if block_format == "start_end":
            # Preserve START/END format
            # Try to extract description from original block
            desc = block_descriptions.get(label, "")
            if not desc:
                # Try to extract from original text
                original_text = block_info.get("text", "")
                desc_match = re.search(
                    rf"{re.escape(comment)}\s*BLOCK\s+{label}\s+START[:\s]*(.*)$",
                    original_text,
                    re.MULTILINE
                )
                if desc_match:
                    desc = desc_match.group(1).strip()

            # Build new block with START/END markers
            separator = "=" * 78
            if desc:
                new_block = (
                    f"{comment}{separator}\n"
                    f"{comment} BLOCK {label} START: {desc}\n"
                    f"{comment}{separator}\n"
                    f"{content}\n"
                    f"{comment} BLOCK {label} END\n"
                )
            else:
                new_block = (
                    f"{comment} BLOCK {label} START\n"
                    f"{content}\n"
                    f"{comment} BLOCK {label} END\n"
                )
        else:
            # Simple format
            new_block = f"{comment} BLOCK {label}\n{content}\n"

        # Replace
        result = result[:start] + new_block + result[end:]

    return result
