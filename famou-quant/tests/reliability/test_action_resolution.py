"""Action resolution: the trajectory must describe what ran, not what was asked.

Not every ExpertKind has an implementation and not every declared family has a
registered expert. The registry substitutes silently, and before this the
DecisionRecord kept the requested label — so the trainer saw two action ids for
one behaviour and spent capacity separating them.
"""

from __future__ import annotations

import pytest

from famou.reliability.experts import default_expert_registry
from famou.reliability.rl.encoding import ActionCodec
from famou.reliability.strategy import ReliabilityAwareQuantStrategy
from famou.reliability.types import ExpertKind, StructuredAction


class _Resolver(ReliabilityAwareQuantStrategy):
    """Just the resolution half — building a full strategy needs a manifest,
    an evaluator and a ledger, none of which this behaviour touches."""

    def __init__(self, families):
        self.experts = default_expert_registry(families=families)
        self.logger = None
        self._substitutions_seen = set()


MIXED = ["formula", "gbdt", "linear", "mlp", "temporal_transformer"]
MODEL = ["gbdt", "linear", "mlp", "temporal_transformer"]


class TestResolution:
    def test_unimplemented_kind_records_what_served_it(self):
        resolver = _Resolver(MIXED)
        action = StructuredAction(expert=ExpertKind.CROSSOVER, model_family="gbdt")

        resolved, expert = resolver._resolve_action(action)

        assert resolved.expert is ExpertKind.CROSSOVER      # request preserved
        assert resolved.resolved_expert == "mutate"          # reality recorded
        assert resolved.was_substituted
        assert expert.kind is ExpertKind.MUTATE

    def test_family_without_an_expert_falls_back_and_says_so(self):
        """`linear` is declared by TaskSpec but has no expert of its own."""
        resolver = _Resolver(MODEL)
        action = StructuredAction(expert=ExpertKind.MUTATE, model_family="linear")

        resolved, expert = resolver._resolve_action(action)

        assert resolved.model_family == "linear"
        assert resolved.resolved_family == "gbdt"
        assert expert.model_family == "gbdt"

    def test_served_action_is_not_marked_substituted(self):
        resolver = _Resolver(MIXED)
        resolved, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt")
        )

        assert not resolved.was_substituted
        assert resolved.effective_expert == "mutate"
        assert resolved.effective_family == "gbdt"

    def test_effective_properties_fall_back_to_the_request(self):
        """An action that never went through resolution still reads sensibly —
        old trajectories have no resolved_* fields."""
        action = StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp")

        assert action.effective_expert == "explore"
        assert action.effective_family == "mlp"
        assert not action.was_substituted

    def test_substitution_is_logged_once_per_distinct_pair(self):
        """A 200-decision run on an unserved family would otherwise bury the
        log, and the useful information is a small fixed set."""
        warnings = []

        class Logger:
            def warning(self, msg):
                warnings.append(msg)

        resolver = _Resolver(MIXED)
        resolver.logger = Logger()
        for _ in range(5):
            resolver._resolve_action(
                StructuredAction(expert=ExpertKind.CROSSOVER, model_family="gbdt")
            )
        resolver._resolve_action(
            StructuredAction(expert=ExpertKind.DEBUG, model_family="formula")
        )

        assert len(warnings) == 2
        assert "crossover/gbdt" in warnings[0]


class TestEncodingUsesWhatRan:
    def test_aliased_actions_encode_identically(self):
        """crossover/gbdt and mutate/gbdt run the same expert, so they must not
        look like two different actions to the trainer."""
        resolver = _Resolver(MIXED)
        codec = ActionCodec()

        asked_crossover, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.CROSSOVER, model_family="gbdt")
        )
        asked_mutate, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt")
        )

        assert codec.encode(asked_crossover) == codec.encode(asked_mutate)

    def test_linear_encodes_as_gbdt(self):
        resolver = _Resolver(MODEL)
        codec = ActionCodec()

        linear, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="linear")
        )
        gbdt, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt")
        )

        assert codec.encode(linear) == codec.encode(gbdt)

    def test_genuinely_distinct_actions_still_differ(self):
        """The point is to merge aliases, not to flatten everything."""
        resolver = _Resolver(MIXED)
        codec = ActionCodec()

        explore, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="formula")
        )
        mutate, _ = resolver._resolve_action(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="formula")
        )

        assert codec.encode(explore) != codec.encode(mutate)

    def test_unresolved_action_encodes_from_the_request(self):
        codec = ActionCodec()
        action = StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp")
        assert codec.encode(action)[0] == codec.encode(
            action.model_copy(update={"resolved_expert": "explore"})
        )[0]


class TestReachableActionSpace:
    def test_reports_only_kinds_the_registry_implements(self):
        space = _Resolver(MIXED).reachable_action_space()
        assert space["experts"] == ["explore", "local_hpo", "mutate"]
        for missing in ("crossover", "debug", "fusion"):
            assert missing not in space["experts"]

    def test_family_without_an_expert_is_not_reachable(self):
        assert "linear" not in _Resolver(MODEL).reachable_action_space()["families"]

    def test_runtime_level_aliases_stay_reachable(self):
        """temporal_transformer has its own expert, so the registry can serve
        it. That its candidates hit the same trainer as mlp is a documented
        choice in famou_candidate_runtime._FAMILIES, not something this layer
        should silently hide."""
        space = _Resolver(MODEL).reachable_action_space()
        assert "temporal_transformer" in space["families"]

    def test_formula_only_run_reports_one_family(self):
        space = _Resolver(["formula"]).reachable_action_space()
        assert space["families"] == ["formula"]


class TestPolicyMasking:
    def _policy(self, tmp_path, *, greedy=True):
        pytest.importorskip("torch")
        import torch

        from famou.reliability.rl.encoding import ENCODING_VERSION
        from famou.reliability.rl.policy import LearnedMetaPolicy
        from famou.reliability.rl.trainer import _build_net

        codec, encoder = ActionCodec(), __import__(
            "famou.reliability.rl.encoding", fromlist=["ObservationEncoder"]
        ).ObservationEncoder()
        net = _build_net(encoder.dim, codec.head_sizes, (16,))
        path = tmp_path / "ck.pt"
        torch.save({
            "state_dict": net.state_dict(), "input_dim": encoder.dim,
            "head_sizes": list(codec.head_sizes), "hidden": [16],
            "encoding_version": ENCODING_VERSION, "policy_version": "test",
        }, path)
        return LearnedMetaPolicy(str(path), greedy=greedy)

    def _obs(self):
        from famou.reliability.observation import AgentObservation

        return AgentObservation(state_version=1, episode_id="E1")

    def test_masked_policy_never_picks_an_unreachable_expert(self, tmp_path):
        # Sampling, not greedy: greedy would pass by picking one legal value
        # every time, which proves nothing about the masked ones.
        policy = self._policy(tmp_path, greedy=False)
        policy.set_reachable_action_space(
            experts=["explore", "mutate", "local_hpo"], families=["gbdt"]
        )

        for _ in range(40):
            action = policy.act(self._obs())
            assert action.expert.value in {"explore", "mutate", "local_hpo"}
            assert action.model_family == "gbdt"

    def test_unmasked_policy_is_unrestricted(self, tmp_path):
        """The mask has to be doing something."""
        policy = self._policy(tmp_path, greedy=False)
        seen = {policy.act(self._obs()).model_family for _ in range(40)}
        assert len(seen) > 1

    def test_masking_everything_is_refused(self, tmp_path):
        """All -inf makes softmax return NaN, which would surface far from
        here as an unexplained crash."""
        policy = self._policy(tmp_path, greedy=False)
        policy.set_reachable_action_space(families=["nonexistent_family"])

        action = policy.act(self._obs())
        assert action.model_family      # some family was still chosen
