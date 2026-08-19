"""Fixtures for the monitor tests.

Fixtures only. Explicit helpers live in ``monitor_artifacts.py``: pytest imports
every ``conftest.py`` under the bare name ``conftest``, so importing helpers from
it collides with ``tests/reliability/conftest.py`` when both suites run in one
invocation. Fixtures are injected by name and have no such problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from monitor_artifacts import build_run_dir


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A minimal but complete experiment directory with one finished rollout."""
    return build_run_dir(tmp_path)
