"""Temporal Leakage Unit Test (§A.3).

The paper states this stage exists and that "only factors with zero leakage are
accepted", but does not publish the rules.  This module supplies them, in two
layers, because neither alone is sufficient.

Layer 1 — static scan
    Pattern-matches the AST for forward-looking constructions: ``shift(-k)``,
    negative ``periods`` on ``diff``/``pct_change``, reversed slices,
    ``iloc[i+k]``, and rolling windows built on a reversed series.  Cheap, exact
    for the idioms it knows, and it produces a message a repair agent can act on.

Layer 2 — truncation probe
    Recompute the factor on a panel truncated at date *T* and compare against the
    full-panel factor restricted to dates ≤ *T*.  A causal factor is *identical*;
    any difference proves information flowed backwards in time.  This is the same
    principle a competition organiser uses when it re-runs your code against a
    truncated table, and it catches what a pattern scan cannot: whole-sample
    normalisation, ``expanding()`` statistics used without a lag, sorting by a
    future column, and any leakage expressed through library calls.

A third, smaller check rides along: **determinism**.  Two runs on identical input
must agree.  A factor that consumes global RNG state is not reproducible, which is
its own defect regardless of causality.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ static scan

#: Methods whose negative argument shifts data backwards in time.
_NEGATIVE_ARG_METHODS = {"shift", "diff", "pct_change", "tshift", "fillna"}

#: Methods that look ahead by construction when used without a positive shift.
_BACKFILL_METHODS = {"bfill", "backfill"}


@dataclass
class LeakageReport:
    """Verdict of the leakage stage."""

    leaked: bool
    findings: List[str] = field(default_factory=list)
    static_findings: List[str] = field(default_factory=list)
    probe_findings: List[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    n_diff_cells: int = 0
    deterministic: bool = True
    probe_ran: bool = False

    @property
    def detail(self) -> str:
        """Findings joined into one line, for the CheckReport detail field."""
        return "; ".join(self.findings) if self.findings else "no leakage detected"

    def to_dict(self) -> Dict[str, Any]:
        """Verdict plus evidence, archived on the LEAKAGE_UNIT_TEST CheckReport."""
        return {
            "leaked": self.leaked,
            "findings": self.findings,
            "static_findings": self.static_findings,
            "probe_findings": self.probe_findings,
            "max_abs_diff": float(self.max_abs_diff),
            "n_diff_cells": int(self.n_diff_cells),
            "deterministic": self.deterministic,
            "probe_ran": self.probe_ran,
        }


def _negative_constant(node: ast.AST) -> bool:
    """True when ``node`` is a literal negative number."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0


def scan_lookahead(code: str) -> List[str]:
    """Static scan for forward-looking constructions.

    Returns a list of human-readable findings; empty means the scan found nothing
    (which is not proof of causality — that is the probe's job).
    """
    findings: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return findings  # the audit stage reports syntax errors

    for node in ast.walk(tree):
        # --- method calls with negative offsets: df['close'].shift(-1)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in _NEGATIVE_ARG_METHODS:
                for arg in node.args:
                    if _negative_constant(arg):
                        findings.append(
                            f"{method}() called with a negative offset -- reads future bars"
                        )
                for kw in node.keywords:
                    if kw.arg in {"periods", "shift", "lag"} and _negative_constant(kw.value):
                        findings.append(
                            f"{method}({kw.arg}=<negative>) -- reads future bars"
                        )
            if method in _BACKFILL_METHODS:
                findings.append(
                    f"{method}() propagates future values backwards into the past"
                )
            if method == "fillna":
                for kw in node.keywords:
                    if (
                        kw.arg == "method"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value in {"bfill", "backfill"}
                    ):
                        findings.append(
                            "fillna(method='bfill') propagates future values backwards"
                        )

        # --- reversed slices: series[::-1], series.iloc[::-1]
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            step = node.slice.step
            if step is not None and _negative_constant(step):
                findings.append("reversed slice [::-1] -- iterates time backwards")

        # --- explicit forward indexing inside a loop body: iloc[i + k]
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr in {"iloc", "values"} and isinstance(node.slice, ast.BinOp):
                if isinstance(node.slice.op, ast.Add):
                    findings.append(
                        f".{node.value.attr}[i + k] -- forward positional index"
                    )

        # --- np.roll with a negative shift
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            pass  # handled above

    # --- textual fallbacks for spellings the AST walk above does not model
    if re.search(r"\.shift\(\s*-\s*\w+\s*\)", code):
        findings.append("shift() with a negative variable offset -- reads future bars")
    if re.search(r"np\.roll\([^)]*,\s*-\s*\d+", code):
        findings.append("np.roll with a negative shift -- rotates future data into the past")
    if re.search(r"iloc\[\s*::\s*-1", code):
        findings.append("iloc[::-1] -- iterates time backwards")

    # De-duplicate while keeping order.
    seen: set[str] = set()
    unique: List[str] = []
    for finding in findings:
        if finding not in seen:
            seen.add(finding)
            unique.append(finding)
    return unique


# --------------------------------------------------------------- truncation probe


def truncation_probe(
    values_full: pd.DataFrame,
    values_truncated: pd.DataFrame,
    cutoff: pd.Timestamp,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> Tuple[bool, float, int, str]:
    """Compare two runs on dates ≤ ``cutoff``.

    Returns ``(leaked, max_abs_diff, n_diff_cells, message)``.

    Comparison is on the intersection of dates and instruments, with both sides
    NaN treated as equal — a factor whose warm-up period is NaN in both runs is
    consistent, not leaky.  A cell that is NaN in one run only *is* a difference:
    it means the presence of future data changed what could be computed.
    """
    if values_full is None or values_truncated is None:
        return False, 0.0, 0, "probe did not run"

    dates = values_full.index[values_full.index <= cutoff]
    dates = dates.intersection(values_truncated.index)
    columns = values_full.columns.intersection(values_truncated.columns)
    if len(dates) == 0 or len(columns) == 0:
        return False, 0.0, 0, "probe window was empty"

    a = values_full.loc[dates, columns].to_numpy(dtype="float64", na_value=np.nan)
    b = values_truncated.loc[dates, columns].to_numpy(dtype="float64", na_value=np.nan)

    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)

    # NaN arithmetic below is intentional -- the masks above decide what counts.
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - b)
        tol = atol + rtol * np.abs(b)
        mismatch = (~both_nan) & (one_nan | (diff > tol))
    n_diff = int(np.nansum(mismatch))
    max_abs = float(np.nanmax(np.where(mismatch & ~one_nan, diff, 0.0))) if n_diff else 0.0

    if n_diff == 0:
        return False, 0.0, 0, "causal: truncation left earlier values unchanged"

    share = n_diff / mismatch.size
    n_nan_only = int((one_nan & ~both_nan).sum())
    message = (
        f"{n_diff} of {mismatch.size} values ({share:.2%}) on dates <= "
        f"{cutoff.date()} changed when later data was removed "
        f"(max abs diff {max_abs:.3g}"
        + (f", {n_nan_only} appeared or vanished" if n_nan_only else "")
        + ") -- the factor uses future information"
    )
    return True, max_abs, n_diff, message



def determinism_check(
    values_a: pd.DataFrame,
    values_b: pd.DataFrame,
    atol: float = 1e-12,
) -> Tuple[bool, str]:
    """Two runs on identical input must agree exactly."""
    if values_a is None or values_b is None:
        return True, "determinism check skipped"
    if values_a.shape != values_b.shape:
        return False, (
            f"two runs produced different shapes {values_a.shape} vs {values_b.shape}"
        )
    a = values_a.to_numpy(dtype="float64", na_value=np.nan)
    b = values_b.to_numpy(dtype="float64", na_value=np.nan)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        mismatch = (~both_nan) & ((np.isnan(a) ^ np.isnan(b)) | (np.abs(a - b) > atol))
    n = int(np.nansum(mismatch))
    if n:
        return False, (
            f"{n} values differed between two identical runs -- the factor is "
            "non-deterministic (unseeded randomness?)"
        )
    return True, "deterministic"


def build_report(
    static_findings: List[str],
    probe: Optional[Tuple[bool, float, int, str]] = None,
    determinism: Optional[Tuple[bool, str]] = None,
) -> LeakageReport:
    """Combine the layers into one verdict."""
    findings = list(static_findings)
    probe_findings: List[str] = []
    leaked = bool(static_findings)
    max_abs = 0.0
    n_diff = 0
    probe_ran = False

    if probe is not None:
        probe_leaked, max_abs, n_diff, message = probe
        probe_ran = True
        if probe_leaked:
            leaked = True
            probe_findings.append(message)
            findings.append(message)

    deterministic = True
    if determinism is not None:
        deterministic, message = determinism
        if not deterministic:
            findings.append(message)

    return LeakageReport(
        leaked=leaked,
        findings=findings,
        static_findings=list(static_findings),
        probe_findings=probe_findings,
        max_abs_diff=max_abs,
        n_diff_cells=n_diff,
        deterministic=deterministic,
        probe_ran=probe_ran,
    )
