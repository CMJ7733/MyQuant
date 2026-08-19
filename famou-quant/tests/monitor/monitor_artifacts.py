"""Artifact factories for the monitor tests.

These live in a uniquely-named module rather than in ``conftest.py`` on purpose.
pytest imports every ``conftest.py`` under the bare module name ``conftest``, so
a test doing ``from conftest import ...`` binds to whichever one was imported
first -- and this repo already has ``tests/reliability/conftest.py``. Running the
two suites in one pytest invocation then fails at collection. Keeping ``conftest``
to fixtures only (which pytest injects by name, no import needed) and putting
explicit helpers here matches the convention the reliability suite already uses.

Every factory writes the *real* on-disk shape Famou produces, so a change to
``LocalStorage`` or ``Program`` that the reader does not follow shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def program(**overrides: Any) -> Dict[str, Any]:
    """One ``programs/<id>.json`` record."""
    record: Dict[str, Any] = {
        "id": "prog_1_0",
        "code": "def solve():\n    return 1\n",
        "generation": 1,
        "iteration": 1,
        "language": "python",
        "file_extension": ".py",
        "parent_id": "init",
        "combined_score": 0.75,
        "validity": 1.0,
        "metrics": {"score": 0.75, "runtime": 0.12},
        "error_info": None,
        "system_prompt": "you evolve code",
        "prompt": "improve this",
        "response": "here you go",
        "created_at": 1_700_000_100.0,
        "meta": {"experiment_id": "exp_demo", "island_id": 0},
    }
    record.update(overrides)
    return record


def rollout(**overrides: Any) -> Dict[str, Any]:
    """One ``results/<rollout_id>.json`` record, in the compact stored form."""
    embedded = overrides.pop("program", program())
    record: Dict[str, Any] = {
        "rollout_id": "rollout_1",
        "experiment_id": "exp_demo",
        "rollout_name": "mutation",
        "iteration": 1,
        "island_id": 0,
        "rollout_attempt": 1,
        "status": "success",
        "failed_module": None,
        "error_message": None,
        "selection": {"parent_id": "init"},
        "generated_program_id": embedded["id"] if embedded else None,
        "program": embedded,
        "stats": {"execution_time": 3.5},
        "llm_request_logs": [llm_request()],
        "created_at": 1_700_000_100.0,
        "completed_at": 1_700_000_103.5,
    }
    record.update(overrides)
    return record


def llm_request(**overrides: Any) -> Dict[str, Any]:
    """One ``llm_requests.log`` line."""
    record: Dict[str, Any] = {
        "request_id": "req-1",
        "model": "gpt-4o",
        "api_base": "https://example.invalid/v1",
        "request_time": "2026-08-16T10:00:00",
        "attempt": 1,
        "max_retries": 3,
        "status": "success",
        "duration_seconds": 2.0,
        "prompt_tokens": 1000,
        "response_tokens": 200,
    }
    record.update(overrides)
    return record


def checkpoint(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "current_iteration": 1,
        "best_program_id": "prog_1_0",
        "best_program_score": 0.75,
    }
    record.update(overrides)
    return record


CONFIG_YAML = """\
experiment:
  name: demo
  strategy: greedy
  language: python
  max_iterations: 10
  task_description: 演示任务
  island:
    num_islands: 2
    population_size: 20
infrastructure:
  llm:
    provider: openai
    model: gpt-4o
  backend:
    mode: threadpool
"""


def build_run_dir(root: Path) -> Path:
    """A minimal but complete experiment directory with one finished rollout."""
    path = root / "exp_demo"
    path.mkdir()
    (path / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

    seed = program(
        id="init",
        generation=0,
        iteration=0,
        parent_id=None,
        combined_score=0.5,
        created_at=1_700_000_000.0,
    )
    write_json(path / "programs" / "init.json", seed)
    (path / "programs" / "init.py").write_text("def solve():\n    return 0\n", encoding="utf-8")

    child = program()
    write_json(path / "programs" / "prog_1_0.json", child)
    (path / "programs" / "prog_1_0.py").write_text(child["code"], encoding="utf-8")

    write_json(path / "results" / "rollout_1.json", rollout())
    write_json(path / "experiment_checkpoint_1.json", checkpoint())
    write_jsonl(path / "llm_requests.log", [llm_request()])
    write_jsonl(
        path / "experiment.jsonl",
        [{"timestamp": 1_700_000_101.0, "level": "INFO", "message": "started"}],
    )
    return path
