from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Literal, Sequence

from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, Action
from fly_crossy.v4.policy import CHANNELS, IMAGE_H, IMAGE_W


ProfileName = Literal["smoke", "80", "1k", "full"]

OBSERVATION_SIZE = OBSERVATION_INPUT_SIZE
FRAME_SHAPE = (IMAGE_H, IMAGE_W, CHANNELS)
ACTION_NAMES = tuple(action.value for action in ACTION_ORDER)

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ProfileBudget:
    name: ProfileName
    training_seeds: int
    validation_seeds: int
    final_test_seeds: int
    dagger_rounds: int
    max_steps: int
    minimum_free_bytes: int


PROFILE_BUDGETS = MappingProxyType(
    {
        "smoke": ProfileBudget("smoke", 2, 3, 3, 1, 32, 2 * _GIB),
        "80": ProfileBudget("80", 64, 21, 100, 4, 200, 2 * _GIB),
        "1k": ProfileBudget("1k", 128, 42, 100, 6, 200, 5 * _GIB),
        "full": ProfileBudget("full", 256, 42, 100, 8, 200, 15 * _GIB),
    }
)


@dataclass(frozen=True, slots=True)
class SeedManifest:
    training: tuple[str, ...]
    validation: tuple[str, ...]
    final_test: tuple[str, ...]

    def validate(self) -> None:
        partitions = (self.training, self.validation, self.final_test)
        if any(not partition for partition in partitions):
            raise ValueError("seed manifest partitions must not be empty")
        if any(len(set(partition)) != len(partition) for partition in partitions):
            raise ValueError("seed manifest partitions must not contain duplicates")
        combined = self.training + self.validation + self.final_test
        if len(set(combined)) != len(combined):
            raise ValueError("seed manifest partitions must be disjoint")

    def sha256(self) -> str:
        self.validate()
        payload = json.dumps(
            {
                "training": self.training,
                "validation": self.validation,
                "finalTest": self.final_test,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def profile_budget(name: str) -> ProfileBudget:
    try:
        return PROFILE_BUDGETS[name]
    except KeyError as error:
        raise ValueError(f"unknown V7 profile: {name}") from error


def build_seed_manifest(
    profile: ProfileName,
    namespace: str = "crossy-v7",
) -> SeedManifest:
    budget = profile_budget(profile)

    def seeds(partition: str, count: int) -> tuple[str, ...]:
        return tuple(
            f"{namespace}-{budget.name}-{partition}-{index:03d}"
            for index in range(count)
        )

    manifest = SeedManifest(
        training=seeds("training", budget.training_seeds),
        validation=seeds("validation", budget.validation_seeds),
        final_test=seeds("final-test", budget.final_test_seeds),
    )
    manifest.validate()
    return manifest


def validate_action_order(actions: Sequence[Action]) -> None:
    if tuple(actions) != ACTION_ORDER:
        raise ValueError("actions must match canonical ACTION_ORDER")
