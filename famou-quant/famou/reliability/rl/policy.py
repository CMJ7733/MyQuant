"""LearnedMetaPolicy — a trained checkpoint in the meta_policy slot.

Drops in wherever ``HeuristicMetaPolicy`` goes, so switching the search from
hand-written rules to a learned policy is one constructor argument.

Three things it does that the heuristic cannot:

- exposes ``last_log_prob`` / ``last_value``, which the strategy records on
  every DecisionRecord. Those two fields were structurally present but always
  None; filling them is what upgrades the trajectory from a cloning corpus to
  something an on-policy correction can use.
- samples rather than argmaxes (temperature-controlled), so a deployed policy
  keeps producing the exploration its next training round needs.
- refuses to load a checkpoint whose encoding version does not match, instead
  of silently feeding the network features that mean something else.

Safety rails that are deliberate, not incidental:

- promotion is MASKED OFF when no sealed budget remains. A learned head has no
  concept of a hard constraint and will happily keep requesting gates; the
  budget ledger would reject them, but every rejected request is a wasted
  decision. Masking is cheaper than teaching.
- promotion targets are resolved from the observation, not invented: the head
  decides *whether* to promote, the observation decides *what* is promotable.
  A hallucinated candidate id would silently fall back to generation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from famou.reliability.observation import AgentObservation
from famou.reliability.rl.encoding import (
    ENCODING_VERSION,
    ActionCodec,
    ObservationEncoder,
)
from famou.reliability.types import Fidelity, StructuredAction


class CheckpointMismatch(RuntimeError):
    """The checkpoint was trained against a different observation space."""


class LearnedMetaPolicy:
    """Meta-controller backed by a trained multi-head network."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        policy_version: Optional[str] = None,
        temperature: float = 1.0,
        greedy: bool = False,
        encoder: Optional[ObservationEncoder] = None,
        codec: Optional[ActionCodec] = None,
        min_f2_seeds_for_promotion: int = 3,
    ):
        import torch

        from famou.reliability.rl.trainer import _build_net

        self.encoder = encoder or ObservationEncoder()
        self.codec = codec or ActionCodec()
        self.temperature = max(1e-3, float(temperature))
        self.greedy = greedy
        self.min_f2_seeds_for_promotion = min_f2_seeds_for_promotion

        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_enc = blob.get("encoding_version")
        if ckpt_enc != ENCODING_VERSION:
            raise CheckpointMismatch(
                f"checkpoint {checkpoint_path} was trained with encoding "
                f"{ckpt_enc!r}, runtime is {ENCODING_VERSION!r}. Retrain "
                "rather than run a policy on features it never saw."
            )
        if blob.get("input_dim") != self.encoder.dim:
            raise CheckpointMismatch(
                f"checkpoint expects {blob.get('input_dim')} features, "
                f"encoder produces {self.encoder.dim}"
            )

        self.net = _build_net(blob["input_dim"], blob["head_sizes"], blob["hidden"])
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()

        self.policy_version = policy_version or blob.get("policy_version", "learned_v0")
        #: read by the strategy when it builds the DecisionRecord
        self.last_log_prob: Optional[float] = None
        self.last_value: Optional[float] = None

    # ------------------------------------------------------------------

    def act(self, obs: AgentObservation) -> StructuredAction:
        import torch

        features = torch.tensor([self.encoder.encode(obs)], dtype=torch.float32)
        with torch.no_grad():
            logits_list, value = self.net(features)

        promo_target = self._promotable(obs)
        indices: List[int] = []
        total_log_prob = 0.0

        for head_index, logits in enumerate(logits_list):
            head_name = self.codec.HEADS[head_index]
            masked = self._mask(head_name, logits[0].clone(), obs, promo_target)
            probs = torch.softmax(masked / self.temperature, dim=-1)
            if self.greedy:
                choice = int(torch.argmax(probs))
            else:
                choice = int(torch.multinomial(probs, num_samples=1))
            indices.append(choice)
            total_log_prob += float(torch.log(probs[choice].clamp_min(1e-12)))

        self.last_log_prob = total_log_prob
        self.last_value = float(value[0])

        action = self.codec.decode(
            indices,
            parent_ids=self._parents(obs, promo_target),
            promotion_target_id=promo_target if indices[-1] == 1 else None,
            rationale=f"{self.policy_version} (logp={total_log_prob:.2f}, "
                      f"v={self.last_value:.3f})",
        )
        return action

    # ------------------------------------------------------------------

    def _mask(self, head: str, logits, obs: AgentObservation, promo_target):
        """Apply hard constraints the network cannot be trusted to respect."""
        import torch

        if head == "promote":
            remaining = (obs.remaining_budget or {}).get("sealed_queries", 0.0)
            if remaining < 1 or promo_target is None:
                logits[1] = float("-inf")   # cannot promote: no budget / no target
        elif head == "fidelity":
            # F0 is a static check, not a search action: a policy that picks it
            # burns an iteration producing no evidence.
            if logits.shape[0] > 0:
                logits[0] = float("-inf")
        return logits

    def _promotable(self, obs: AgentObservation) -> Optional[str]:
        """The best candidate that could legitimately be gated right now.

        The network decides whether to spend a sealed query; which candidate it
        applies to is read off the observation, so the policy cannot name a
        candidate that does not exist or has already been gated.
        """
        for cand in obs.top_visible_evidence:
            if (
                cand.n_f2_seeds >= self.min_f2_seeds_for_promotion
                and not cand.certified
                and cand.gate_attempts == 0
            ):
                return cand.candidate_id
        return None

    @staticmethod
    def _parents(obs: AgentObservation, promo_target: Optional[str]) -> List[str]:
        if promo_target:
            return [promo_target]
        if obs.certified_candidates:
            return [obs.certified_candidates[0].candidate_id]
        if obs.top_visible_evidence:
            return [obs.top_visible_evidence[0].candidate_id]
        return []

    # ------------------------------------------------------------------

    def explain(self, obs: AgentObservation) -> Dict[str, Any]:
        """Per-head probabilities + the features behind them.

        A policy that spends GPU hours should be inspectable; this is what a
        paper's qualitative analysis section is written from.
        """
        import torch

        features = torch.tensor([self.encoder.encode(obs)], dtype=torch.float32)
        with torch.no_grad():
            logits_list, value = self.net(features)
        out: Dict[str, Any] = {"value": float(value[0]),
                               "features": self.encoder.describe(obs)}
        from famou.reliability.rl.encoding import (
            BATCH_BUCKETS, EXPERTS, FAMILIES, FIDELITIES,
        )

        vocab = {"expert": EXPERTS, "family": FAMILIES,
                 "fidelity": [str(f) for f in FIDELITIES],
                 "batch": [str(b) for b in BATCH_BUCKETS],
                 "promote": ["no", "yes"]}
        for i, head in enumerate(self.codec.HEADS):
            probs = torch.softmax(logits_list[i][0] / self.temperature, dim=-1)
            out[head] = {
                name: round(float(p), 4)
                for name, p in zip(vocab[head], probs)
            }
        return out
