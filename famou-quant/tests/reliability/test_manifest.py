"""Manifest loading from the frozen Protocol B splits files."""

from __future__ import annotations

from pathlib import Path

import pytest

from famou.reliability.types import FrozenSplitManifest

SPLITS_V2 = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "evoquant_qlib"
    / "protocol_b"
    / "splits_v2.yaml"
)


@pytest.mark.skipif(not SPLITS_V2.exists(), reason="protocol_b splits not present")
class TestFrozenManifest:
    def test_load_e11_post_cutoff(self):
        m = FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E11")
        assert m.episode_role == "evaluation_post_cutoff"
        assert m.train.start == "2018-01-01"
        assert m.visible_dev.start == "2024-01-01"
        assert m.sealed_promotion.start == "2025-01-01"
        assert m.final_test.end == "2026-08-10"
        assert m.embargo_days == 2
        assert m.data_sha256 is not None

    def test_load_e1_development(self):
        m = FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E1")
        assert m.episode_role == "development"

    def test_hash_stable_and_episode_specific(self):
        a = FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E11")
        b = FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E11")
        c = FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E2")
        assert a.compute_hash() == b.compute_hash()  # deterministic
        assert a.compute_hash() != c.compute_hash()  # episode-specific

    def test_unknown_episode_raises(self):
        with pytest.raises(KeyError):
            FrozenSplitManifest.from_yaml(str(SPLITS_V2), "E99")
