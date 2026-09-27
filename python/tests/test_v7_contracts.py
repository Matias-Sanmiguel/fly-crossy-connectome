from __future__ import annotations

from dataclasses import replace

import pytest

from fly_crossy.schema import ACTION_ORDER
from fly_crossy.v7.contracts import (
    ACTION_NAMES,
    FRAME_SHAPE,
    OBSERVATION_SIZE,
    SeedManifest,
    build_seed_manifest,
    profile_budget,
    validate_action_order,
)


GIB = 1024**3


@pytest.mark.parametrize(
    ("name", "training", "validation", "final_test", "rounds", "steps", "disk"),
    (
        ("smoke", 2, 3, 3, 1, 32, 2 * GIB),
        ("80", 64, 21, 100, 4, 200, 2 * GIB),
        ("1k", 128, 42, 100, 6, 200, 5 * GIB),
        ("full", 256, 42, 100, 8, 200, 15 * GIB),
    ),
)
def test_profile_budget_enforces_evidence_and_resource_limits(
    name: str,
    training: int,
    validation: int,
    final_test: int,
    rounds: int,
    steps: int,
    disk: int,
) -> None:
    budget = profile_budget(name)

    assert budget.name == name
    assert budget.training_seeds == training
    assert budget.validation_seeds == validation
    assert budget.final_test_seeds == final_test
    assert budget.dagger_rounds == rounds
    assert budget.max_steps == steps
    assert budget.minimum_free_bytes == disk


def test_v7_io_contract_matches_canonical_environment() -> None:
    assert OBSERVATION_SIZE == 517
    assert FRAME_SHAPE == (24, 48, 3)
    assert ACTION_NAMES == ("forward", "backward", "left", "right", "wait")
    validate_action_order(ACTION_ORDER)


def test_profile_budget_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="profile"):
        profile_budget("500")


def test_action_contract_rejects_reordered_actions() -> None:
    with pytest.raises(ValueError, match="canonical ACTION_ORDER"):
        validate_action_order(tuple(reversed(ACTION_ORDER)))


def test_seed_manifest_is_deterministic_disjoint_and_namespaced() -> None:
    first = build_seed_manifest("smoke", namespace="unit-v7")
    second = build_seed_manifest("smoke", namespace="unit-v7")

    assert first == second
    assert first.sha256() == second.sha256()
    assert first.training == (
        "unit-v7-smoke-training-000",
        "unit-v7-smoke-training-001",
    )
    assert first.validation[0] == "unit-v7-smoke-validation-000"
    assert first.final_test[-1] == "unit-v7-smoke-final-test-002"
    assert len(set(first.training + first.validation + first.final_test)) == 8


@pytest.mark.parametrize(
    "manifest",
    (
        SeedManifest((), ("validation",), ("test",)),
        SeedManifest(("duplicate", "duplicate"), ("validation",), ("test",)),
        SeedManifest(("overlap",), ("overlap",), ("test",)),
        SeedManifest(("training",), ("overlap",), ("overlap",)),
    ),
)
def test_seed_manifest_rejects_empty_duplicate_or_overlapping_partitions(
    manifest: SeedManifest,
) -> None:
    with pytest.raises(ValueError, match="seed"):
        manifest.validate()


def test_seed_manifest_hash_changes_with_partition_content() -> None:
    manifest = build_seed_manifest("smoke", namespace="unit-v7")
    changed = replace(
        manifest,
        training=("unit-v7-smoke-training-999", *manifest.training[1:]),
    )

    assert manifest.sha256() != changed.sha256()
