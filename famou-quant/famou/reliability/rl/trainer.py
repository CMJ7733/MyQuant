"""Offline training for the meta-policy: stage A (BC) and stage B (AWR).

Both stages train the same factored multi-head network, which is what lets
stage B start from stage A's weights instead of from noise.

Stage A — Behaviour Cloning
    Supervised: predict the heuristic's action from the observation. It cannot
    beat the heuristic by construction; the point is to reach heuristic-level
    competence before anything stochastic happens, so that later exploration
    perturbs a sensible policy rather than a random one.

Stage B — Advantage-Weighted Regression
    Same cross-entropy loss, each sample weighted by ``exp(advantage / beta)``
    where the advantage comes from the delayed reward and a learned value
    head. AWR was chosen over Q-learning variants deliberately: it never
    bootstraps off its own Q estimates, so it cannot diverge on the few
    hundred transitions this system realistically produces. Its failure mode
    is staying too close to the data — a safe failure for a policy that spends
    real compute.

Weights are clipped: one lucky transition with a huge reward would otherwise
dominate the entire update.

Torch is imported lazily so the search loop does not depend on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from famou.reliability.rl.encoding import (
    ENCODING_VERSION,
    ActionCodec,
    ObservationEncoder,
)
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import PolicyCheckpoint


class TrainingDataError(RuntimeError):
    """Not enough usable transitions to train on."""


@dataclass
class Sample:
    features: List[float]
    action: Tuple[int, ...]
    reward: Optional[float] = None
    ret: Optional[float] = None          # discounted return-to-go
    decision_id: str = ""
    policy_version: str = ""
    state_version: int = 0               # used to detect run boundaries


@dataclass
class TrainingReport:
    stage: str
    n_samples: int
    epochs: int
    losses: List[float] = field(default_factory=list)
    head_accuracy: Dict[str, float] = field(default_factory=dict)
    mean_reward: Optional[float] = None
    value_loss: Optional[float] = None
    notes: str = ""

    def summary(self) -> str:
        acc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.head_accuracy.items()))
        tail = f" final_loss={self.losses[-1]:.4f}" if self.losses else ""
        return f"[{self.stage}] n={self.n_samples} epochs={self.epochs}{tail} acc({acc})"


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


def build_dataset(
    store: TrajectoryStore,
    *,
    codec: Optional[ActionCodec] = None,
    gamma: float = 0.9,
    require_reward: bool = False,
    encoding_version: str = ENCODING_VERSION,
) -> List[Sample]:
    """Join DecisionRecords with their Transitions into training samples.

    Records written by an older encoder are skipped rather than mixed in:
    feeding a policy features whose meaning has shifted is worse than training
    on less data.
    """
    codec = codec or ActionCodec()
    decisions = store.decisions()
    transitions = store.transitions()

    by_decision = {t.decision_ref: t for t in transitions}
    samples: List[Sample] = []
    skipped_version = skipped_features = skipped_reward = 0

    # Trajectory order = commit order, which is what makes return-to-go
    # meaningful; transitions() preserves append order.
    ordered = [t for t in transitions if t.decision_ref in decisions]
    for t in ordered:
        d = decisions[t.decision_ref]
        if d.encoding_version and d.encoding_version != encoding_version:
            skipped_version += 1
            continue
        if not d.observation_features:
            skipped_features += 1
            continue
        if require_reward and t.reward is None:
            skipped_reward += 1
            continue
        samples.append(
            Sample(
                features=list(d.observation_features),
                action=codec.encode(d.structured_action),
                reward=t.reward,
                decision_id=d.decision_id,
                policy_version=d.policy_version,
                state_version=d.state_version,
            )
        )

    # Return-to-go, computed WITHIN each run. Rewards are already delayed and
    # attributed per transition, so this only propagates credit backwards a few
    # steps — enough to prefer decisions that set up a later promotion.
    #
    # Run boundaries matter once trajectories from several search runs are
    # concatenated, which is exactly what iterated offline training does. Each
    # run starts from a fresh StateStore, so its state_version restarts near 0;
    # a version that does not advance marks a new run. Without this split, the
    # last decision of run A would collect credit from the first decision of
    # run B — a reward it could not possibly have caused.
    for segment in _split_runs(samples):
        running = 0.0
        for s in reversed(segment):
            if s.reward is not None:
                running = s.reward + gamma * running
                s.ret = running
            else:
                s.ret = None

    return samples


def _split_runs(samples: List[Sample]) -> List[List[Sample]]:
    """Partition an ordered sample list into per-run segments."""
    if not samples:
        return []
    segments: List[List[Sample]] = [[samples[0]]]
    for prev, cur in zip(samples, samples[1:]):
        if cur.state_version <= prev.state_version:
            segments.append([cur])          # version restarted -> new run
        else:
            segments[-1].append(cur)
    return segments


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


def _build_net(input_dim: int, head_sizes: Sequence[int], hidden: Sequence[int]):
    import torch.nn as nn

    class MultiHeadPolicy(nn.Module):
        """Shared trunk (the 'belief encoder') + one head per action factor,
        plus a value head used by AWR."""

        def __init__(self):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleList([nn.Linear(prev, n) for n in head_sizes])
            self.value = nn.Linear(prev, 1)

        def forward(self, x):
            z = self.trunk(x)
            return [head(z) for head in self.heads], self.value(z).squeeze(-1)

    return MultiHeadPolicy()


class _TrainerBase:
    def __init__(
        self,
        *,
        encoder: Optional[ObservationEncoder] = None,
        codec: Optional[ActionCodec] = None,
        hidden: Sequence[int] = (128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ):
        self.encoder = encoder or ObservationEncoder()
        self.codec = codec or ActionCodec()
        self.hidden = tuple(hidden)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.net = None

    # -- helpers --------------------------------------------------------

    def _tensors(self, samples: List[Sample]):
        import torch

        x = torch.tensor([s.features for s in samples], dtype=torch.float32)
        y = torch.tensor([list(s.action) for s in samples], dtype=torch.long)
        return x, y

    def _init_net(self):
        import torch

        torch.manual_seed(self.seed)
        self.net = _build_net(self.encoder.dim, self.codec.head_sizes, self.hidden)
        return self.net

    def _accuracy(self, logits_list, y) -> Dict[str, float]:
        import torch

        out: Dict[str, float] = {}
        with torch.no_grad():
            for i, name in enumerate(self.codec.HEADS):
                pred = logits_list[i].argmax(dim=-1)
                out[name] = float((pred == y[:, i]).float().mean())
        return out

    def save(self, path: str, *, policy_version: str, notes: str = "",
             trained_on: int = 0) -> PolicyCheckpoint:
        """Persist weights + the encoding contract they assume."""
        import torch

        if self.net is None:
            raise RuntimeError("nothing trained yet")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_dim": self.encoder.dim,
                "head_sizes": list(self.codec.head_sizes),
                "hidden": list(self.hidden),
                "encoding_version": ENCODING_VERSION,
                "policy_version": policy_version,
            },
            target,
        )
        meta = PolicyCheckpoint(
            policy_version=policy_version,
            kind="bc" if isinstance(self, BehaviorCloning) else "offline_rl",
            artifact_uri=str(target),
            trained_on_transitions=trained_on,
            notes=notes,
        )
        target.with_suffix(".meta.json").write_text(
            json.dumps(meta.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        return meta


class BehaviorCloning(_TrainerBase):
    """Stage A: imitate whatever policy produced the trajectories."""

    def fit(
        self,
        samples: List[Sample],
        *,
        epochs: int = 200,
        batch_size: int = 64,
        min_samples: int = 8,
    ) -> TrainingReport:
        import torch
        import torch.nn.functional as F

        if len(samples) < min_samples:
            raise TrainingDataError(
                f"behaviour cloning needs >= {min_samples} samples, got {len(samples)}"
            )
        net = self._init_net()
        x, y = self._tensors(samples)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        losses: List[float] = []
        n = x.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits_list, _ = net(x[idx])
                loss = sum(
                    F.cross_entropy(logits_list[h], y[idx, h])
                    for h in range(len(self.codec.HEADS))
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item() * len(idx)
            losses.append(epoch_loss / n)

        net.eval()
        with torch.no_grad():
            logits_list, _ = net(x)
        return TrainingReport(
            stage="behaviour_cloning",
            n_samples=n,
            epochs=epochs,
            losses=losses,
            head_accuracy=self._accuracy(logits_list, y),
            notes="stage A: cannot exceed the demonstrator by construction",
        )


class AdvantageWeightedRegression(_TrainerBase):
    """Stage B: reweight cloning by how much better than average an action did.

    ``beta`` is the temperature: small beta trusts the advantage estimate and
    moves further from the data; large beta degenerates to plain cloning. The
    default is conservative on purpose — with a few hundred transitions the
    advantage estimate is noisy, and an over-confident update spends real GPU
    hours on a worse policy.
    """

    def fit(
        self,
        samples: List[Sample],
        *,
        epochs: int = 200,
        batch_size: int = 64,
        beta: float = 0.5,
        max_weight: float = 20.0,
        value_epochs: int = 100,
        init_from: Optional["_TrainerBase"] = None,
        min_samples: int = 8,
    ) -> TrainingReport:
        import torch
        import torch.nn.functional as F

        usable = [s for s in samples if s.ret is not None]
        if len(usable) < min_samples:
            raise TrainingDataError(
                f"AWR needs >= {min_samples} reward-labelled samples, "
                f"got {len(usable)} (has RewardBuilder.mature() run?)"
            )

        if init_from is not None and init_from.net is not None:
            # Warm start from behaviour cloning — the whole reason both stages
            # share one architecture.
            self.net = init_from.net
        else:
            self._init_net()
        net = self.net

        x, y = self._tensors(usable)
        returns = torch.tensor([s.ret for s in usable], dtype=torch.float32)
        # Normalise: returns live on whatever scale RewardConfig chose, and the
        # exp() below is very sensitive to that scale.
        r_mean, r_std = returns.mean(), returns.std().clamp_min(1e-6)
        returns_n = (returns - r_mean) / r_std

        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        # --- fit the value head first: advantages need a baseline ---------
        v_loss = None
        for _ in range(value_epochs):
            _, values = net(x)
            loss = F.mse_loss(values, returns_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            v_loss = loss.detach().item()

        # --- advantage-weighted policy update -----------------------------
        with torch.no_grad():
            _, baseline = net(x)
            adv = returns_n - baseline
            weights = torch.exp(adv / beta).clamp(max=max_weight)
            # A single outlier reward must not become the entire gradient.
            weights = weights / weights.mean().clamp_min(1e-6)

        losses: List[float] = []
        n = x.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits_list, _ = net(x[idx])
                loss = sum(
                    (F.cross_entropy(logits_list[h], y[idx, h], reduction="none")
                     * weights[idx]).mean()
                    for h in range(len(self.codec.HEADS))
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item() * len(idx)
            losses.append(epoch_loss / n)

        net.eval()
        with torch.no_grad():
            logits_list, _ = net(x)
        return TrainingReport(
            stage="awr",
            n_samples=n,
            epochs=epochs,
            losses=losses,
            head_accuracy=self._accuracy(logits_list, y),
            mean_reward=float(returns.mean()),
            value_loss=v_loss,
            notes=f"beta={beta}, weights clipped at {max_weight}",
        )
