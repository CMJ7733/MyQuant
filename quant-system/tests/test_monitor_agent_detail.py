"""Per-agent detail aggregation for the monitor reader."""

from __future__ import annotations

import json
import os
from pathlib import Path

from cogalpha.agents.hierarchy import HIERARCHY
from cogalpha.monitor.reader import RunReader
from conftest import call, generation, write_jsonl


def test_known_agent_detail_has_static_identity_before_run_starts(run_dir: Path) -> None:
    detail = RunReader(run_dir).agent_detail("AgentMarketCycle")

    assert detail is not None
    assert set(detail) == {
        "name",
        "display_name",
        "level",
        "layer",
        "focus",
        "probe",
        "selected",
        "status",
        "current_generation",
        "current_cycle",
        "summary",
        "trajectory",
        "generations",
        "recent_operations",
    }
    assert detail["name"] == "AgentMarketCycle"
    assert detail["display_name"] == "MarketCycle"
    assert detail["level"] == 1
    assert detail["layer"] == "Market Structure & Cycle Layer"
    assert "Long-term trends" in detail["focus"]
    market_cycle = next(spec for spec in HIERARCHY if spec.name == "AgentMarketCycle")
    assert detail["probe"] == market_cycle.probe
    assert detail["selected"] is False
    assert detail["status"] == "queued"
    assert detail["current_generation"] is None
    assert detail["current_cycle"] is None
    assert detail["summary"] == {
        "generations": 0,
        "generated": 0,
        "passed": 0,
        "qualified": 0,
        "elite": 0,
        "best_score": None,
        "best_rank_ic": None,
        "llm_calls": 0,
        "llm_tokens": 0,
        "mean_latency_ms": 0.0,
        "seconds": 0.0,
        "stopped_early": None,
    }
    assert detail["trajectory"] == []
    assert detail["generations"] == []
    assert detail["recent_operations"] == []


def test_unknown_agent_detail_returns_none(run_dir: Path) -> None:
    assert RunReader(run_dir).agent_detail("AgentNotInHierarchy") is None


def test_agent_detail_exposes_identity_for_all_21_hierarchy_agents(run_dir: Path) -> None:
    reader = RunReader(run_dir)

    assert len(HIERARCHY) == 21
    for spec in HIERARCHY:
        detail = reader.agent_detail(spec.name)

        assert detail is not None
        assert {
            "name": detail["name"],
            "level": detail["level"],
            "layer": detail["layer"],
            "focus": detail["focus"],
            "probe": detail["probe"],
        } == {
            "name": spec.name,
            "level": spec.level,
            "layer": spec.layer,
            "focus": spec.focus,
            "probe": spec.probe,
        }


def test_agent_detail_aggregates_only_the_requested_agent(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(generation=0, cycle=0, n_qualified=3, n_elite=2),
            generation(generation=1, cycle=1, n_qualified=2, n_elite=1),
            generation(agent="AgentTailRisk", generation=0, cycle=0),
        ],
    )
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [
            call(seq=1, agent="AgentMarketCycle", tokens=10),
            call(
                seq=2,
                agent="AgentMarketCycle",
                tokens=20,
                generation=1,
                cycle=0,
                role="judge",
                mode="light",
                temperature=0.7,
                model="mock-model",
                response="response-2",
            ),
            call(seq=3, agent="AgentTailRisk", tokens=999),
        ],
    )

    reader = RunReader(run_dir)
    reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["selected"] is True
    assert detail["status"] == "done"
    assert detail["current_generation"] == 1
    assert detail["current_cycle"] == 1
    assert detail["summary"] == {
        "generations": 2,
        "generated": 14,
        "passed": 8,
        "qualified": 5,
        "elite": 3,
        "best_score": 0.086,
        "best_rank_ic": 0.071,
        "llm_calls": 2,
        "llm_tokens": 30,
        "mean_latency_ms": 1.5,
        "seconds": 25.0,
        "stopped_early": None,
    }
    assert detail["trajectory"] == [
        {"generation": 0, "cycle": 0, "score": 0.072, "elite": 2},
        {"generation": 1, "cycle": 1, "score": 0.072, "elite": 1},
    ]
    assert [item["generation"] for item in detail["generations"]] == [0, 1]
    assert set(detail["generations"][0]) == {
        "generation",
        "cycle",
        "generated",
        "passed",
        "qualified",
        "elite",
        "elite_mean_score",
        "best",
        "reject_counts",
        "op_counts",
        "llm_calls",
        "wall_seconds",
    }
    assert detail["generations"][0]["wall_seconds"] == 12.5
    assert [item["seq"] for item in detail["recent_operations"]] == [2, 1]
    assert detail["recent_operations"][0] == {
        "seq": 2,
        "role": "judge",
        "agent": "AgentMarketCycle",
        "generation": 1,
        "cycle": 0,
        "mode": "light",
        "temperature": 0.7,
        "model": "mock-model",
        "tokens": 20,
        "latency_ms": 2,
        "chars": len("response-2"),
    }
    assert all(
        item["agent"] == "AgentMarketCycle" for item in detail["recent_operations"]
    )
    assert all("prompt" not in item for item in detail["recent_operations"])
    assert all("response" not in item for item in detail["recent_operations"])


def test_agent_detail_caps_recent_operations_without_losing_totals(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=seq, agent="AgentMarketCycle", tokens=1) for seq in range(1, 206)],
    )

    reader = RunReader(run_dir)
    reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["summary"]["llm_calls"] == 205
    assert detail["summary"]["llm_tokens"] == 205
    assert len(detail["recent_operations"]) == 200
    assert detail["recent_operations"][0]["seq"] == 205
    assert detail["recent_operations"][-1]["seq"] == 6


def test_agent_detail_falls_back_to_generation_calls_and_rounds_digest_seconds(
    run_dir: Path,
) -> None:
    best = {"name": "alpha-seven", "score": 0.091, "rank_ic": 0.073}
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(
                generation=3,
                cycle=2,
                op_counts={"generate": 5, "mutate": 4},
                n_generated=9,
                n_passed_checker=6,
                n_qualified=4,
                n_elite=2,
                reject_counts={"judge": 3},
                best=best,
                elite_mean_score=0.081,
                llm_calls=7,
                wall_seconds=12.56,
            )
        ],
    )

    reader = RunReader(run_dir)
    reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["summary"]["llm_calls"] == 7
    assert detail["summary"]["llm_tokens"] == 0
    assert detail["summary"]["mean_latency_ms"] == 0.0
    assert detail["generations"] == [
        {
            "generation": 3,
            "cycle": 2,
            "generated": 9,
            "passed": 6,
            "qualified": 4,
            "elite": 2,
            "elite_mean_score": 0.081,
            "best": best,
            "reject_counts": {"judge": 3},
            "op_counts": {"generate": 5, "mutate": 4},
            "llm_calls": 7,
            "wall_seconds": 12.6,
        }
    ]


def test_shrunk_archive_replacement_rebuilds_both_streams(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "generations.jsonl",
        [generation(generation=0), generation(generation=1)],
    )
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=1, tokens=10), call(seq=2, tokens=20)],
    )
    reader = RunReader(run_dir)
    reader.poll()
    old_generation_size = (run_dir / "generations.jsonl").stat().st_size
    old_call_size = (run_dir / "llm_calls.jsonl").stat().st_size

    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(
                agent="AgentTailRisk",
                generation=4,
                n_generated=2,
                n_passed_checker=1,
                n_qualified=1,
                n_elite=0,
                llm_calls=1,
            )
        ],
    )
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=101, agent="AgentTailRisk", tokens=5)],
    )
    assert (run_dir / "generations.jsonl").stat().st_size < old_generation_size
    assert (run_dir / "llm_calls.jsonl").stat().st_size < old_call_size

    snapshot = reader.poll()
    market = reader.agent_detail("AgentMarketCycle")
    tail = reader.agent_detail("AgentTailRisk")

    assert snapshot.generations_seen == 1
    assert snapshot.totals["llm_calls"] == 1
    assert snapshot.totals["llm_tokens"] == 5
    assert [item["seq"] for item in snapshot.recent_calls] == [101]
    assert market is not None
    assert market["selected"] is False
    assert market["status"] == "queued"
    assert market["summary"]["generations"] == 0
    assert market["summary"]["llm_calls"] == 0
    assert market["summary"]["llm_tokens"] == 0
    assert market["generations"] == []
    assert market["recent_operations"] == []
    assert tail is not None
    assert tail["current_generation"] == 4
    assert tail["summary"]["generations"] == 1
    assert tail["summary"]["generated"] == 2
    assert tail["summary"]["llm_calls"] == 1
    assert tail["summary"]["llm_tokens"] == 5
    assert [item["seq"] for item in tail["recent_operations"]] == [101]


def test_same_size_in_place_replacement_starts_a_new_epoch(run_dir: Path) -> None:
    generation_path = run_dir / "generations.jsonl"
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(generation_path, [generation(generation=1, n_generated=7)])
    write_jsonl(call_path, [call(seq=1, tokens=20)])
    reader = RunReader(run_dir)
    reader.poll()
    generation_stat = generation_path.stat()
    call_stat = call_path.stat()

    write_jsonl(generation_path, [generation(generation=2, n_generated=8)])
    write_jsonl(call_path, [call(seq=2, tokens=22)])
    assert generation_path.stat().st_size == generation_stat.st_size
    assert call_path.stat().st_size == call_stat.st_size
    os.utime(
        generation_path,
        ns=(generation_stat.st_atime_ns, generation_stat.st_mtime_ns + 1_000_000_000),
    )
    os.utime(
        call_path,
        ns=(call_stat.st_atime_ns, call_stat.st_mtime_ns + 1_000_000_000),
    )

    snapshot = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert snapshot.generations_seen == 1
    assert snapshot.totals["llm_calls"] == 1
    assert snapshot.totals["llm_tokens"] == 22
    assert detail is not None
    assert detail["current_generation"] == 2
    assert detail["summary"]["generations"] == 1
    assert detail["summary"]["generated"] == 8
    assert detail["summary"]["llm_calls"] == 1
    assert detail["summary"]["llm_tokens"] == 22
    assert [item["seq"] for item in detail["recent_operations"]] == [2]


def test_regrown_in_place_replacement_starts_a_new_epoch(run_dir: Path) -> None:
    generation_path = run_dir / "generations.jsonl"
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(generation_path, [generation(generation=1, n_generated=7)])
    write_jsonl(call_path, [call(seq=1, tokens=20)])
    reader = RunReader(run_dir)
    reader.poll()
    old_generation_size = generation_path.stat().st_size
    old_call_size = call_path.stat().st_size

    write_jsonl(
        generation_path,
        [
            generation(agent="AgentTailRisk", generation=8, n_generated=9),
            generation(agent="AgentTailRisk", generation=9, n_generated=10),
        ],
    )
    write_jsonl(
        call_path,
        [
            call(seq=21, agent="AgentTailRisk", tokens=21),
            call(seq=22, agent="AgentTailRisk", tokens=22),
        ],
    )
    assert generation_path.stat().st_size > old_generation_size
    assert call_path.stat().st_size > old_call_size

    snapshot = reader.poll()
    market = reader.agent_detail("AgentMarketCycle")
    tail = reader.agent_detail("AgentTailRisk")

    assert snapshot.generations_seen == 2
    assert snapshot.totals["llm_calls"] == 2
    assert snapshot.totals["llm_tokens"] == 43
    assert market is not None
    assert market["summary"]["generations"] == 0
    assert tail is not None
    assert tail["summary"]["generations"] == 2
    assert tail["summary"]["generated"] == 19
    assert [item["seq"] for item in tail["recent_operations"]] == [22, 21]


def test_repeated_poll_without_file_changes_is_idempotent(run_dir: Path) -> None:
    write_jsonl(run_dir / "generations.jsonl", [generation(generation=3)])
    write_jsonl(run_dir / "llm_calls.jsonl", [call(seq=9, tokens=17)])
    reader = RunReader(run_dir)

    first = reader.poll()
    second = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert first.generations_seen == second.generations_seen == 1
    assert first.totals == second.totals
    assert first.recent_calls == second.recent_calls
    assert detail is not None
    assert detail["summary"]["generations"] == 1
    assert detail["summary"]["llm_calls"] == 1
    assert detail["summary"]["llm_tokens"] == 17
    assert [item["seq"] for item in detail["recent_operations"]] == [9]


def test_ordinary_append_preserves_existing_epoch(run_dir: Path) -> None:
    generation_path = run_dir / "generations.jsonl"
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(generation_path, [generation(generation=0, n_generated=7)])
    write_jsonl(call_path, [call(seq=1, tokens=10)])
    reader = RunReader(run_dir)
    reader.poll()

    generation_path.write_text(
        generation_path.read_text(encoding="utf-8")
        + json.dumps(generation(generation=1, n_generated=8))
        + "\n",
        encoding="utf-8",
    )
    call_path.write_text(
        call_path.read_text(encoding="utf-8")
        + json.dumps(call(seq=2, tokens=20))
        + "\n",
        encoding="utf-8",
    )

    snapshot = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert snapshot.generations_seen == 2
    assert snapshot.totals["llm_calls"] == 2
    assert snapshot.totals["llm_tokens"] == 30
    assert detail is not None
    assert detail["summary"]["generated"] == 15
    assert [item["seq"] for item in detail["recent_operations"]] == [2, 1]


def test_partial_trailing_line_is_applied_once_when_completed(run_dir: Path) -> None:
    generation_path = run_dir / "generations.jsonl"
    encoded = json.dumps(generation(generation=5))
    split = len(encoded) // 2
    generation_path.write_text(encoded[:split], encoding="utf-8")
    reader = RunReader(run_dir)

    first = reader.poll()
    generation_path.write_text(encoded + "\n", encoding="utf-8")
    second = reader.poll()
    third = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert first.generations_seen == 0
    assert second.generations_seen == third.generations_seen == 1
    assert detail is not None
    assert detail["current_generation"] == 5
    assert detail["summary"]["generations"] == 1


def test_unknown_call_tags_do_not_allocate_per_agent_state(run_dir: Path) -> None:
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=seq, agent=f"AgentUnknown{seq}", tokens=1) for seq in range(1, 101)],
    )
    reader = RunReader(run_dir)

    snapshot = reader.poll()

    assert len(snapshot.recent_calls) == 60
    assert reader._agent_recent == {}
    assert reader._agent_call_counts == {}
    assert reader._agent_token_counts == {}
    assert reader._agent_latency_totals == {}


def test_first_canonical_call_marks_agent_running_with_active_context(
    run_dir: Path,
) -> None:
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=1, agent="AgentMarketCycle", generation=4, cycle=2)],
    )
    reader = RunReader(run_dir)

    snapshot = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert snapshot.generations_seen == 0
    assert snapshot.current_agent == "AgentMarketCycle"
    assert snapshot.current_generation == 4
    assert snapshot.current_cycle == 2
    assert detail is not None
    assert detail["selected"] is True
    assert detail["status"] == "running"
    assert detail["current_generation"] == 4
    assert detail["current_cycle"] == 2


def test_snapshot_agent_call_count_increases_across_polls(run_dir: Path) -> None:
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(call_path, [call(seq=1, agent="AgentMarketCycle")])
    reader = RunReader(run_dir)

    first = reader.poll()
    first_agent = next(
        agent for agent in first.agents if agent["name"] == "AgentMarketCycle"
    )

    with call_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(call(seq=2, agent="AgentMarketCycle")) + "\n")

    second = reader.poll()
    second_agent = next(
        agent for agent in second.agents if agent["name"] == "AgentMarketCycle"
    )

    assert first_agent["llm_calls"] == 1
    assert second_agent["llm_calls"] == 2


def test_snapshot_agent_call_count_uses_generation_or_observed_max(
    run_dir: Path,
) -> None:
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(
        run_dir / "generations.jsonl",
        [generation(agent="AgentMarketCycle", llm_calls=7)],
    )
    write_jsonl(call_path, [call(seq=1, agent="AgentMarketCycle")])
    reader = RunReader(run_dir)

    first = reader.poll()
    with call_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(call(seq=2, agent="AgentMarketCycle")) + "\n")
    second = reader.poll()

    first_agent = next(
        agent for agent in first.agents if agent["name"] == "AgentMarketCycle"
    )
    second_agent = next(
        agent for agent in second.agents if agent["name"] == "AgentMarketCycle"
    )
    assert first_agent["llm_calls"] == 7
    assert second_agent["llm_calls"] == 7
    assert reader.agent_detail("AgentMarketCycle")["summary"]["llm_calls"] == 7


def test_canonical_call_transition_finishes_previous_without_reviving_it(
    run_dir: Path,
) -> None:
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(
        call_path,
        [
            call(seq=1, agent="AgentMarketCycle", generation=2, cycle=0),
            call(seq=2, agent="AgentTailRisk", generation=0, cycle=1),
        ],
    )
    reader = RunReader(run_dir)

    transitioned = reader.poll()
    market = reader.agent_detail("AgentMarketCycle")
    tail = reader.agent_detail("AgentTailRisk")

    assert transitioned.current_agent == "AgentTailRisk"
    assert transitioned.current_generation == 0
    assert transitioned.current_cycle == 1
    assert market is not None and market["status"] == "done"
    assert tail is not None and tail["status"] == "running"

    # A delayed call from the completed agent is still accounted for, but is not
    # a strong enough signal to undo the established sequential transition.
    with call_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(call(seq=3, agent="AgentMarketCycle", generation=2, cycle=0))
            + "\n"
        )

    after_delayed_call = reader.poll()

    assert after_delayed_call.current_agent == "AgentTailRisk"
    assert after_delayed_call.current_generation == 0
    assert after_delayed_call.current_cycle == 1
    assert reader.agent_detail("AgentMarketCycle")["status"] == "done"
    assert reader.agent_detail("AgentMarketCycle")["summary"]["llm_calls"] == 2
    assert reader.agent_detail("AgentTailRisk")["status"] == "running"


def test_unknown_call_does_not_disturb_running_canonical_agent(run_dir: Path) -> None:
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(
        call_path,
        [call(seq=1, agent="AgentMarketCycle", generation=5, cycle=2)],
    )
    reader = RunReader(run_dir)
    reader.poll()

    with call_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(call(seq=2, agent="AgentUnknown", generation=99)) + "\n")

    snapshot = reader.poll()

    assert snapshot.current_agent == "AgentMarketCycle"
    assert snapshot.current_generation == 5
    assert snapshot.current_cycle == 2
    assert "AgentUnknown" not in reader._agents
    assert reader.agent_detail("AgentMarketCycle")["status"] == "running"


def test_generation_transition_takes_priority_over_delayed_completed_agent_call(
    run_dir: Path,
) -> None:
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(agent="AgentMarketCycle", generation=3, cycle=0),
            generation(agent="AgentTailRisk", generation=1, cycle=2),
        ],
    )
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [call(seq=8, agent="AgentMarketCycle", generation=3, cycle=0)],
    )
    reader = RunReader(run_dir)

    snapshot = reader.poll()

    assert snapshot.current_agent == "AgentTailRisk"
    assert snapshot.current_generation == 1
    assert snapshot.current_cycle == 2
    assert reader.agent_detail("AgentMarketCycle")["status"] == "done"
    assert reader.agent_detail("AgentMarketCycle")["summary"]["llm_calls"] == 1
    assert reader.agent_detail("AgentTailRisk")["status"] == "running"


def test_delayed_generation_across_polls_does_not_reverse_call_transition(
    run_dir: Path,
) -> None:
    generation_path = run_dir / "generations.jsonl"
    write_jsonl(
        run_dir / "llm_calls.jsonl",
        [
            call(seq=1, agent="AgentMarketCycle", generation=3, cycle=0),
            call(seq=2, agent="AgentTailRisk", generation=0, cycle=1),
        ],
    )
    reader = RunReader(run_dir)
    transitioned = reader.poll()
    assert transitioned.current_agent == "AgentTailRisk"

    with generation_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                generation(
                    agent="AgentMarketCycle",
                    generation=3,
                    cycle=0,
                    n_generated=9,
                )
            )
            + "\n"
        )

    after_delayed_generation = reader.poll()
    market = reader.agent_detail("AgentMarketCycle")
    tail = reader.agent_detail("AgentTailRisk")

    assert after_delayed_generation.current_agent == "AgentTailRisk"
    assert after_delayed_generation.current_generation == 0
    assert after_delayed_generation.current_cycle == 1
    assert market is not None and market["status"] == "done"
    assert market["summary"]["generations"] == 1
    assert market["summary"]["generated"] == 9
    assert market["current_generation"] == 3
    assert tail is not None and tail["status"] == "running"


def test_agent_detail_uses_generation_call_total_when_call_log_is_partial(
    run_dir: Path,
) -> None:
    write_jsonl(run_dir / "generations.jsonl", [generation(llm_calls=7)])
    write_jsonl(run_dir / "llm_calls.jsonl", [call(seq=1, tokens=10)])
    reader = RunReader(run_dir)

    reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert detail is not None
    assert detail["summary"]["llm_calls"] == 7
    assert detail["summary"]["llm_tokens"] == 10
    assert detail["summary"]["mean_latency_ms"] == 1.0


def test_mutating_agent_detail_does_not_change_later_results(run_dir: Path) -> None:
    best = {"name": "alpha-original", "score": 0.086, "rank_ic": 0.071}
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            generation(
                best=best,
                reject_counts={"judge": 3},
                op_counts={"generate": 7},
            )
        ],
    )
    write_jsonl(run_dir / "llm_calls.jsonl", [call(seq=1, tokens=10)])
    reader = RunReader(run_dir)
    reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")
    assert detail is not None

    detail["recent_operations"][0]["tokens"] = 999
    detail["generations"][0]["best"]["score"] = 999
    detail["generations"][0]["reject_counts"]["judge"] = 999
    detail["generations"][0]["op_counts"]["generate"] = 999
    later = reader.agent_detail("AgentMarketCycle")

    assert later is not None
    assert later["recent_operations"][0]["tokens"] == 10
    assert later["generations"][0]["best"]["score"] == 0.086
    assert later["generations"][0]["reject_counts"] == {"judge": 3}
    assert later["generations"][0]["op_counts"] == {"generate": 7}


def test_replacement_between_preflight_and_consumption_rebuilds_atomically(
    run_dir: Path, monkeypatch
) -> None:
    generation_path = run_dir / "generations.jsonl"
    call_path = run_dir / "llm_calls.jsonl"
    write_jsonl(generation_path, [generation(generation=1)])
    write_jsonl(call_path, [call(seq=1, tokens=10)])
    reader = RunReader(run_dir)
    reader.poll()

    next_generation_path = run_dir / "generations.next"
    next_call_path = run_dir / "llm_calls.next"
    write_jsonl(
        next_generation_path,
        [generation(agent="AgentTailRisk", generation=7, n_generated=2, llm_calls=1)],
    )
    write_jsonl(
        next_call_path,
        [call(seq=71, agent="AgentTailRisk", tokens=7)],
    )
    original_poll = reader._gen_tail.poll
    replaced = False

    def replace_then_consume():
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(next_generation_path, generation_path)
            os.replace(next_call_path, call_path)
        return original_poll()

    monkeypatch.setattr(reader._gen_tail, "poll", replace_then_consume)

    snapshot = reader.poll()
    market = reader.agent_detail("AgentMarketCycle")
    tail = reader.agent_detail("AgentTailRisk")

    assert snapshot.generations_seen == 1
    assert snapshot.totals["llm_calls"] == 1
    assert snapshot.totals["llm_tokens"] == 7
    assert [item["seq"] for item in snapshot.recent_calls] == [71]
    assert market is not None
    assert market["summary"]["generations"] == 0
    assert market["summary"]["llm_calls"] == 0
    assert tail is not None
    assert tail["current_generation"] == 7
    assert tail["summary"]["generations"] == 1
    assert tail["summary"]["generated"] == 2
    assert [item["seq"] for item in tail["recent_operations"]] == [71]


def test_mutating_snapshot_recent_calls_does_not_change_later_results(
    run_dir: Path,
) -> None:
    write_jsonl(run_dir / "llm_calls.jsonl", [call(seq=1, tokens=10)])
    reader = RunReader(run_dir)
    snapshot = reader.poll()

    snapshot.recent_calls[0]["tokens"] = 999
    later_snapshot = reader.poll()
    detail = reader.agent_detail("AgentMarketCycle")

    assert later_snapshot.recent_calls[0]["tokens"] == 10
    assert detail is not None
    assert detail["recent_operations"][0]["tokens"] == 10
