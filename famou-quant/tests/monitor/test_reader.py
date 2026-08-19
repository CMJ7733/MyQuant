"""Reader tests: incremental ingest, snapshot arithmetic, detail lookups."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from monitor_artifacts import checkpoint, llm_request, program, rollout, write_json, write_jsonl
from famou.monitor.reader import ExperimentReader


def test_snapshot_reports_config_and_totals(run_dir: Path) -> None:
    state = ExperimentReader(run_dir).poll()

    assert state.experiment_id == "exp_demo"
    assert state.config["strategy"] == "greedy"
    assert state.config["num_islands"] == 2
    assert state.max_iterations == 10
    assert state.current_iteration == 1
    assert state.progress == pytest.approx(0.1)
    # The seed has no rollout behind it, so counting rollouts would miss it. This is
    # the same count report.py:_load_evolve_data reports.
    assert state.totals["n_programs"] == 2
    assert state.totals["n_rollouts"] == 1
    assert state.totals["best_score"] == pytest.approx(0.75)
    # The one attempt appears both in llm_requests.log and in the rollout's own copy,
    # because the framework writes the rollout's entries into that same log.
    assert state.totals["llm_calls"] == 1
    assert state.totals["prompt_tokens"] == 1000


def test_the_same_llm_attempt_is_not_counted_twice(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "llm_requests.log",
        [
            llm_request(),                                  # duplicate of the rollout's copy
            llm_request(request_id="req-1", attempt=2),      # a retry of the same call
            llm_request(request_id="req-2"),                 # a different call
        ],
    )

    state = ExperimentReader(run_dir).poll()

    assert state.totals["llm_calls"] == 3
    assert state.llm_by_model[0]["calls"] == 3


def test_islands_match_population_stats(run_dir: Path) -> None:
    state = ExperimentReader(run_dir).poll()

    assert len(state.islands) == 1
    island = state.islands[0]
    assert island["island_id"] == 0
    assert island["n_programs"] == 2
    assert island["best_score"] == pytest.approx(0.75)
    assert island["avg_score"] == pytest.approx(0.625)
    assert island["max_generation"] == 1
    assert island["n_error"] == 0


def test_buggy_program_counts_as_error(run_dir: Path) -> None:
    write_json(
        run_dir / "programs" / "prog_1_1.json",
        program(id="prog_1_1", validity=0.0, combined_score=0.0),
    )

    state = ExperimentReader(run_dir).poll()

    assert state.islands[0]["n_error"] == 1


def test_funnel_counts_each_stage(run_dir: Path) -> None:
    write_json(
        run_dir / "results" / "rollout_2.json",
        rollout(rollout_id="rollout_2", status="failed", failed_module="evaluate", program=None),
    )

    state = ExperimentReader(run_dir).poll()
    counts = {stage["key"]: stage["count"] for stage in state.funnel}

    assert counts == {
        "dispatched": 2,
        "generated": 1,
        "evaluated": 1,
        "valid": 1,
        "improved": 1,
    }
    assert state.reject_counts == [{"reason": "evaluate", "count": 1}]


def test_checkpoint_best_score_wins_over_running_maximum(run_dir: Path) -> None:
    # The framework's own verdict is authoritative even when a later program scored
    # higher in a rollout that has not been checkpointed yet.
    write_json(run_dir / "experiment_checkpoint_2.json", checkpoint(best_program_score=0.9, best_program_id="x"))

    state = ExperimentReader(run_dir).poll()

    assert state.totals["best_score"] == pytest.approx(0.9)
    assert state.totals["best_program_id"] == "x"


def test_newest_checkpoint_is_chosen_numerically(run_dir: Path) -> None:
    # "10" must beat "9"; a lexicographic sort would pick the wrong file.
    write_json(run_dir / "experiment_checkpoint_9.json", checkpoint(current_iteration=9))
    write_json(run_dir / "experiment_checkpoint_10.json", checkpoint(current_iteration=10))

    assert ExperimentReader(run_dir).poll().current_iteration == 10


def test_poll_ingests_artifacts_written_after_construction(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    reader.poll()

    write_json(run_dir / "programs" / "prog_2_0.json", program(id="prog_2_0", iteration=2, combined_score=0.9))
    write_json(
        run_dir / "results" / "rollout_2.json",
        rollout(rollout_id="rollout_2", iteration=2, program=program(id="prog_2_0", combined_score=0.9)),
    )
    state = reader.poll()

    assert state.totals["n_programs"] == 3
    assert state.totals["n_rollouts"] == 2
    assert [point["best_score"] for point in state.best_trajectory] == [0.75, 0.9]


def test_each_artifact_is_ingested_only_once(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    reader.poll()
    # Touch the directory so the mtime shortcut does not mask a double ingest.
    write_json(run_dir / "results" / "rollout_2.json", rollout(rollout_id="rollout_2"))

    state = reader.poll()

    assert state.totals["n_rollouts"] == 2
    assert state.totals["n_programs"] == 2


def test_directory_reset_rebuilds_from_scratch(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    assert reader.poll().totals["n_rollouts"] == 1

    # A new run reusing the directory: the old rollout is gone, a different one is
    # there. Folding the new one on top of the old state would double-count.
    (run_dir / "results" / "rollout_1.json").unlink()
    write_json(run_dir / "results" / "rollout_9.json", rollout(rollout_id="rollout_9"))

    state = reader.poll()

    assert state.totals["n_rollouts"] == 1
    assert state.recent_rollouts[0]["rollout_id"] == "rollout_9"


def test_truncated_jsonl_rebuilds_from_scratch(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    reader.poll()

    write_jsonl(run_dir / "llm_requests.log", [llm_request(request_id="req-2")])
    state = reader.poll()

    # The rewritten log's one call, plus the distinct one in the rollout's copy.
    assert state.totals["llm_calls"] == 2


def test_partial_trailing_line_is_completed_on_the_next_poll(run_dir: Path) -> None:
    path = run_dir / "llm_requests.log"
    reader = ExperimentReader(run_dir)
    assert reader.poll().totals["llm_calls"] == 1

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "req-9", "model": "gpt-4o", "status": "suc')
    assert reader.poll().totals["llm_calls"] == 1  # the half-written line is not counted

    with path.open("a", encoding="utf-8") as handle:
        handle.write('cess", "prompt_tokens": 5}\n')
    assert reader.poll().totals["llm_calls"] == 2


def test_warnings_surface_newest_first(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "experiment.jsonl",
        [
            {"timestamp": 1.0, "level": "INFO", "message": "quiet"},
            {"timestamp": 2.0, "level": "WARNING", "message": "island reset"},
            {"timestamp": 3.0, "level": "ERROR", "message": "evaluate crashed"},
        ],
    )

    state = ExperimentReader(run_dir).poll()

    assert [w["level"] for w in state.warnings] == ["ERROR", "WARNING"]


def test_live_is_false_for_an_old_run(run_dir: Path) -> None:
    assert ExperimentReader(run_dir).poll().live is False


def test_a_completed_llm_call_counts_as_liveness(run_dir: Path) -> None:
    """A slow LLM call is the only evidence a run is alive while it runs.

    Nothing is written between dispatching a rollout and its program landing, so on
    a run whose slowest call took 390s the dashboard declared a healthy run stopped.
    The log stamps ISO strings rather than the epoch floats every other stream uses,
    which is why this needs its own conversion.
    """
    import datetime

    now = datetime.datetime.now()
    write_jsonl(
        run_dir / "llm_requests.log",
        [llm_request(request_id="req-fresh", request_time=now.isoformat())],
    )

    state = ExperimentReader(run_dir).poll()

    assert state.live is True
    assert state.last_event_at == pytest.approx(now.timestamp(), abs=1.0)


def test_an_unparseable_llm_timestamp_is_ignored(run_dir: Path) -> None:
    """A malformed time must not crash the poll or fake liveness."""
    write_jsonl(
        run_dir / "llm_requests.log",
        [llm_request(request_id="req-bad", request_time="not-a-timestamp")],
    )

    state = ExperimentReader(run_dir).poll()

    assert state.live is False
    assert state.totals["llm_calls"] == 2  # still counted for cost


def test_stale_after_default_covers_a_slow_llm_call() -> None:
    """Guard the measured threshold.

    180s was inherited from cogalpha, where a generation takes 30-60s. Here a single
    call was measured at 390s, so that default reported a healthy run as stopped.
    """
    import inspect

    default = inspect.signature(ExperimentReader.__init__).parameters["stale_after"].default
    assert default >= 390.0


def test_live_is_true_for_a_fresh_event(run_dir: Path) -> None:
    now = time.time()
    write_json(run_dir / "results" / "rollout_2.json", rollout(rollout_id="rollout_2", completed_at=now))

    assert ExperimentReader(run_dir).poll().live is True


def test_finished_when_the_iteration_budget_is_spent(run_dir: Path) -> None:
    write_json(run_dir / "experiment_checkpoint_2.json", checkpoint(current_iteration=10))

    assert ExperimentReader(run_dir).poll().finished is True


def test_program_detail_reads_code_and_lineage(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    reader.poll()

    detail = reader.program_detail("prog_1_0")

    assert detail is not None
    assert "def solve" in detail["code"]
    assert detail["prompt"] == "improve this"
    assert [ancestor["id"] for ancestor in detail["lineage"]] == ["prog_1_0", "init"]


def test_program_detail_falls_back_to_the_sibling_source_file(run_dir: Path) -> None:
    record = program(id="prog_2_0", file_extension=".h")
    record.pop("code")
    write_json(run_dir / "programs" / "prog_2_0.json", record)
    (run_dir / "programs" / "prog_2_0.h").write_text("// header", encoding="utf-8")

    detail = ExperimentReader(run_dir).program_detail("prog_2_0")

    assert detail is not None and detail["code"] == "// header"


def test_lineage_survives_a_parent_cycle(run_dir: Path) -> None:
    write_json(run_dir / "programs" / "a.json", program(id="a", parent_id="b"))
    write_json(run_dir / "programs" / "b.json", program(id="b", parent_id="a"))
    reader = ExperimentReader(run_dir)
    reader.poll()

    assert [ancestor["id"] for ancestor in reader.program_detail("a")["lineage"]] == ["a", "b"]


def test_rollout_detail_drops_the_duplicated_program_dump(run_dir: Path) -> None:
    detail = ExperimentReader(run_dir).rollout_detail("rollout_1")

    assert detail is not None
    assert "program" not in detail
    assert detail["generated_program_id"] == "prog_1_0"
    assert detail["stats"]["execution_time"] == 3.5


def test_island_and_iteration_lookups(run_dir: Path) -> None:
    reader = ExperimentReader(run_dir)
    reader.poll()

    assert [p["id"] for p in reader.island_detail(0)["programs"]] == ["init", "prog_1_0"]
    assert [p["id"] for p in reader.iteration_programs(1)] == ["prog_1_0"]
    assert reader.island_detail(7) is None


def test_resolve_follows_the_newest_experiment_under_a_parent(run_dir: Path) -> None:
    older = run_dir.parent / "exp_old"
    older.mkdir()
    (older / "config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    import os

    os.utime(older, (1, 1))

    assert ExperimentReader(run_dir.parent).path == run_dir


def test_missing_directory_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Start the run first"):
        ExperimentReader(tmp_path / "nope")


def test_parent_without_any_experiment_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.yaml or programs/"):
        ExperimentReader(tmp_path)
