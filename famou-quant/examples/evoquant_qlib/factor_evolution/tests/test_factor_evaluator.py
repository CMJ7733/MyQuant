"""Tests for the factor evaluator.

The look-ahead tests are the important ones. The evaluator hands LLM-written
code the raw price panel, so a candidate that reads future rows is inevitable,
and such a factor scores spectacularly while being worthless. These tests pin
down that the causality probe actually catches it — including the tempting
cases where the leak is small or hidden behind a NaN pattern.

Run:
    /opt/conda/envs/quant/bin/python -m pytest examples/evoquant_qlib/factor_evolution/tests/ -v
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from evaluator import SPLIT_CFG, evaluate  # noqa: E402
from panel import load_panel  # noqa: E402


def test_the_default_episode_is_the_development_one() -> None:
    """Guard the sandbox default.

    E1 is the development episode -- already burned under v1, so it may be
    looked at as often as the method is still changing. An evaluation episode's
    only value is that nobody has looked yet, so defaulting to one would spend
    it by accident. Switching is deliberate: FAMOU_EPISODE=E11.
    """
    assert SPLIT_CFG["episode"] == "E1"
    assert SPLIT_CFG["role"] == "development"


def write_candidate(tmp_path: Path, body: str) -> str:
    """Materialise a candidate module and return its path."""
    path = tmp_path / "candidate.py"
    path.write_text("import numpy as np\n\n" + textwrap.dedent(body), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------- honest

def test_seed_factor_scores(tmp_path: Path) -> None:
    result = evaluate(str(HERE.parent / "seed_factor.py"))

    assert result["validity"] == 1.0, result.get("error_info")
    assert result["combined_score"] == result["rank_ic"]
    assert result["episode"] == SPLIT_CFG["episode"]
    # Every scoreable dev day, not a magic number: the count changes with the
    # episode, and hardcoding it would fail the moment FAMOU_EPISODE is switched.
    panel = load_panel(SPLIT_CFG)
    assert result["n_ic_days"] <= panel.n_days - panel.dev_start_row
    assert result["n_ic_days"] > 200          # a year of trading, minus embargo
    for key in ("ic", "icir", "sharpe", "long_short_return"):
        assert isinstance(result[key], float)
    assert len(result["subperiod_rank_ic"]) == 4


def test_an_honest_multi_field_factor_is_accepted(tmp_path: Path) -> None:
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            close, volume = panel.close, panel.volume
            out = np.full_like(close, np.nan)
            # Uses several fields and two windows, all strictly backward-looking.
            out[20:] = -(close[20:] / close[:-20] - 1.0) * np.log1p(volume[20:])
            return out
    """)

    result = evaluate(program)

    assert result["validity"] == 1.0, result.get("error_info")


# -------------------------------------------------------------------- cheating

def test_blatant_look_ahead_is_rejected(tmp_path: Path) -> None:
    """The factor IS tomorrow's return. Without the probe this scores ~1.0."""
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            close = panel.close
            out = np.full_like(close, np.nan)
            out[:-2] = close[2:] / close[1:-1] - 1.0     # the label itself
            return out
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert result["combined_score"] == 0.0
    assert "looks ahead" in result["error_info"]


def test_subtle_one_day_look_ahead_is_rejected(tmp_path: Path) -> None:
    """A single row of leak is still a leak, and is easy to write by accident."""
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            close = panel.close
            out = np.full_like(close, np.nan)
            # Off-by-one: row t uses close[t+1].
            out[20:-1] = -(close[21:] / close[1:-20] - 1.0)
            return out
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "looks ahead" in result["error_info"]


def test_full_sample_statistic_is_rejected(tmp_path: Path) -> None:
    """Normalising by a whole-window mean leaks the future into every row.

    This is the leak most likely to be written innocently -- it looks like
    ordinary standardisation, and nothing about it mentions the future.
    """
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            close = panel.close
            out = np.full_like(close, np.nan)
            out[20:] = -(close[20:] / close[:-20] - 1.0)
            # np.nanmean over axis 0 spans the entire window, including days
            # after t.
            return out - np.nanmean(out, axis=0, keepdims=True)
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "looks ahead" in result["error_info"]


def test_look_ahead_hidden_in_the_nan_pattern_is_rejected(tmp_path: Path) -> None:
    """Values match, but a cell is only revealed when the future says so.

    Comparing numbers alone would pass this, which is why the probe compares
    the NaN masks too.
    """
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            close = panel.close
            out = np.full_like(close, np.nan)
            out[20:] = -(close[20:] / close[:-20] - 1.0)
            if out.shape[0] > 3:
                # Blank out any cell whose stock rises tomorrow.
                future_up = np.zeros_like(out, dtype=bool)
                future_up[:-1] = close[1:] > close[:-1]
                out[future_up] = np.nan
            return out
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "looks ahead" in result["error_info"]


# --------------------------------------------------------------------- broken

def test_wrong_shape_is_rejected(tmp_path: Path) -> None:
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            return panel.close[:, :5]
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "shape" in result["error_info"]


def test_constant_factor_is_rejected(tmp_path: Path) -> None:
    """A constant expresses no ranking; IC would be undefined, not zero."""
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            return np.ones_like(panel.close)
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "constant" in result["error_info"]


def test_all_nan_factor_is_rejected(tmp_path: Path) -> None:
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            return np.full_like(panel.close, np.nan)
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0


def test_crashing_candidate_is_reported_not_raised(tmp_path: Path) -> None:
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            raise ValueError("boom")
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "boom" in result["error_info"]


def test_missing_entry_point_is_reported(tmp_path: Path) -> None:
    program = write_candidate(tmp_path, """
        def some_other_name(panel):
            return panel.close
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "compute_factor" in result["error_info"]


def test_timeout_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAMOU_EVAL_TIMEOUT", "5")
    program = write_candidate(tmp_path, """
        def compute_factor(panel):
            while True:
                pass
    """)

    result = evaluate(program)

    assert result["validity"] == 0.0
    assert "exceeded" in result["error_info"]


# -------------------------------------------------------- score cannot be faked

def test_a_candidate_cannot_report_its_own_score(tmp_path: Path) -> None:
    """Printing a FAMOU_RESULT line must not influence the score.

    Scoring lives in the parent process precisely so that the only way to score
    well is to produce a factor that predicts returns.
    """
    program = write_candidate(tmp_path, """
        print('FAMOU_RESULT {"combined_score": 9.99, "validity": 1.0}')

        def compute_factor(panel):
            close = panel.close
            out = np.full_like(close, np.nan)
            out[20:] = -(close[20:] / close[:-20] - 1.0)
            return out
    """)

    result = evaluate(program)

    assert result["combined_score"] != 9.99
    assert abs(result["combined_score"]) < 1.0
