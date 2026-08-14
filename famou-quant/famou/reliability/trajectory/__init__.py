"""Transition construction and the Trajectory Store.

Every completed WorkBatch produces one Transition linking:
    observation digest -> decision -> candidates -> evidence -> verdict -> cost

The store is append-only JSONL so the RL trainer can stream it without
loading the whole experiment checkpoint.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from famou.reliability.types import (
    DecisionRecord,
    EvaluationCost,
    GateVerdict,
    Transition,
)


def build_transition(
    *,
    decision: DecisionRecord,
    candidate_ids: List[str],
    evidence_ids: List[str],
    costs: EvaluationCost,
    gate_verdict: Optional[GateVerdict] = None,
    done: bool = False,
) -> Transition:
    return Transition(
        transition_id=f"tr_{uuid.uuid4().hex[:12]}",
        decision_ref=decision.decision_id,
        observation_digest=decision.observation_digest,
        action=decision.structured_action,
        evidence_refs=evidence_ids,
        candidate_ids=candidate_ids,
        gate_verdict=gate_verdict,
        reward=None,  # delayed: filled by the reward builder when it matures
        costs=costs,
        done=done,
        state_version=decision.state_version,
        policy_version=decision.policy_version,
    )


class TrajectoryStore:
    """Append-only trajectory log (JSONL) + in-memory index.

    Thread-safe for appends. ``finalize_rewards`` is called by the reward
    builder once delayed outcomes (gate verdicts, certified-archive gains)
    are known.
    """

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._transitions: List[Transition] = []
        self._decisions: Dict[str, DecisionRecord] = {}
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("kind") == "decision":
                    d = DecisionRecord.model_validate(rec["payload"])
                    self._decisions[d.decision_id] = d
                elif rec.get("kind") == "transition":
                    self._transitions.append(Transition.model_validate(rec["payload"]))

    def _append(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"kind": kind, "payload": payload},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record_decision(self, decision: DecisionRecord) -> None:
        with self._lock:
            self._decisions[decision.decision_id] = decision
            self._append("decision", decision.model_dump(mode="json"))

    def record_transition(self, transition: Transition) -> None:
        with self._lock:
            self._transitions.append(transition)
            self._append("transition", transition.model_dump(mode="json"))

    def transitions(self) -> List[Transition]:
        with self._lock:
            return list(self._transitions)

    def decisions(self) -> Dict[str, DecisionRecord]:
        with self._lock:
            return dict(self._decisions)

    def set_reward(self, transition_id: str, reward: float) -> bool:
        """Fill in a delayed reward (rewrites the in-memory copy and appends
        an update record; JSONL stays append-only)."""
        with self._lock:
            for t in self._transitions:
                if t.transition_id == transition_id:
                    t.reward = reward
                    self._append(
                        "reward_update",
                        {"transition_id": transition_id, "reward": reward},
                    )
                    return True
            return False
