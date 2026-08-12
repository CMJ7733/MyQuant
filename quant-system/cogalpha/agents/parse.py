"""Extract alpha functions from LLM output.

An LLM returns prose plus fenced code, sometimes several factors per response,
sometimes a bare function with no fence, sometimes a fence labelled ``py`` or
nothing at all.  This module turns any of that into a list of :class:`Alpha`
objects, and rejects anything that is not a single well-formed function — badly
shaped output is a *generation* failure and should be counted as one, not
crash the loop.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

from cogalpha.types import Alpha, Lineage

_FENCE_RE = re.compile(r"```(?:python|py|Python)?\s*\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)


class ParseError(ValueError):
    """Raised when a response contains no usable alpha function."""


def extract_blocks(text: str) -> List[str]:
    """Return candidate code blocks from a response, best effort.

    Fenced blocks win.  Failing that, if the whole response looks like code
    (contains a top-level ``def``), it is treated as one block — models
    occasionally drop the fence and the code is otherwise fine.
    """
    blocks = [m.group(1) for m in _FENCE_RE.finditer(text)]
    if blocks:
        return [b for b in (b.strip() for b in blocks) if b]
    if _DEF_RE.search(text):
        return [text.strip()]
    return []


def split_functions(block: str) -> List[str]:
    """Split a block containing several top-level functions into one each.

    Module-level imports preceding the first ``def`` are prepended to every
    function, since a model that writes ``import numpy as np`` at the top expects
    it to be in scope.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        # Unparseable: hand the block back whole so the checker can report the
        # syntax error against the actual text the model produced.
        return [block]

    lines = block.splitlines()
    preamble: List[str] = []
    funcs: List[Tuple[int, int]] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(block, node)
            if seg:
                preamble.append(seg)
        elif isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            # Include decorators, if any, and run to the end of the node.
            if node.decorator_list:
                start = min(start, node.decorator_list[0].lineno - 1)
            end = getattr(node, "end_lineno", None)
            if end is None:  # pragma: no cover - py<3.8 only
                continue
            funcs.append((start, end))

    if len(funcs) <= 1:
        return [block]

    head = "\n".join(preamble)
    out: List[str] = []
    for start, end in funcs:
        body = "\n".join(lines[start:end])
        out.append(f"{head}\n\n{body}".strip() if head else body)
    return out


def function_name(code: str) -> Optional[str]:
    """Name of the single top-level function, or ``None`` if not exactly one."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        match = _DEF_RE.search(code)
        return match.group(1) if match else None
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return names[0] if len(names) == 1 else None


def extract_docstring(code: str) -> str:
    """First docstring in the code, flattened to a single paragraph."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            if doc:
                return " ".join(doc.split())
    return ""


def parse_alphas(
    text: str,
    lineage: Lineage,
    max_alphas: Optional[int] = None,
) -> List[Alpha]:
    """Parse every alpha in ``text`` into :class:`Alpha` objects.

    Syntactically broken candidates are *kept* (with the name recovered by regex
    where possible) so the quality checker can attempt a repair — discarding them
    here would hide the repair loop the paper considers part of the method.
    """
    alphas: List[Alpha] = []
    seen: set[str] = set()

    for block in extract_blocks(text):
        for code in split_functions(block):
            code = code.strip()
            if not code:
                continue
            name = function_name(code)
            if name is None:
                match = _DEF_RE.search(code)
                if match is None:
                    # No function at all: nothing a repair agent could work with.
                    continue
                name = match.group(1)
            if not name.startswith("factor"):
                name = f"factor_{name}"
            alpha = Alpha(
                code=code,
                name=name,
                rationale=extract_docstring(code),
                lineage=Lineage(**{**lineage.to_dict(), "op": lineage.op}),
            )
            if alpha.alpha_id in seen:
                continue
            seen.add(alpha.alpha_id)
            alphas.append(alpha)
            if max_alphas is not None and len(alphas) >= max_alphas:
                return alphas
    return alphas


def rename_function(code: str, new_name: str) -> str:
    """Rename the single top-level function and its self-referential column name.

    Generated code habitually uses the function name as the output column name
    (``out['factor_x'] = ...``); a rename that misses the string literal produces
    a KeyError at return time.
    """
    old = function_name(code)
    if old is None or old == new_name:
        return code
    code = re.sub(rf"\bdef\s+{re.escape(old)}\s*\(", f"def {new_name}(", code, count=1)
    code = code.replace(f"'{old}'", f"'{new_name}'").replace(f'"{old}"', f'"{new_name}"')
    return code


def parse_verdict(text: str, pass_token: str = "PASS") -> Tuple[bool, str]:
    """Read a ``VERDICT: X`` line from a reviewer response.

    Missing verdict is treated as a failure with the raw text as the reason: a
    reviewer that did not answer the question has not approved anything.
    """
    match = re.search(r"VERDICT\s*:\s*([A-Za-z_]+)", text)
    if match is None:
        head = " ".join(text.split())[:300]
        return False, f"no VERDICT line in reviewer response: {head}"
    verdict = match.group(1).strip().upper()
    detail = text[match.end() :].strip()
    return verdict == pass_token.upper(), " ".join(detail.split())[:1000]
