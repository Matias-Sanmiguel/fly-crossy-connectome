from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.dagger_collect import collect_dagger_round
from fly_crossy.models import SettledPopulationFixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import _build_controller_v2_graph


def _make_student(path: Path) -> None:
    graph = _build_controller_v2_graph(
        load_reduced_graph_variant("1k")
    )
    model = SettledPopulationFixedGraphPolicy(
        graph,
        OBSERVATION_INPUT_SIZE,
        len(ACTION_ORDER),
    )
    checkpoint = {
        "format_version": 1,
        "environment_version": 11,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": graph.to_checkpoint(),
            "interface": "population-settled",
        },
        "model_state_dict": model.state_dict(),
        "training": {
            "seed": "dagger-student-test",
            "steps": 2,
            "envs": 1,
            "learning_rate": 3e-4,
            "world_seeds": [
                "dagger-student-test:0000",
                "dagger-student-test:0001",
            ],
            "connectome_graph": "1k",
            "connectome_interface": "population-settled",
            "training_mode": "test",
            "internal_steps": 2,
        },
    }
    torch.save(checkpoint, path)


def _make_base(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    observations = np.zeros(
        (2, OBSERVATION_INPUT_SIZE),
        dtype=np.float32,
    )
    actions = np.asarray([0, 0], dtype=np.int64)
    offsets = np.asarray([0, 2], dtype=np.int64)
    seeds = np.asarray(["base-test:0000"])
    scores = np.asarray([1.0], dtype=np.float32)
    terminals = np.asarray([""])

    dataset_path = path / "dataset.npz"
    np.savez(
        dataset_path,
        observations=observations,
        actions=actions,
        episode_offsets=offsets,
        seeds=seeds,
        final_scores=scores,
        terminal_reasons=terminals,
    )
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "version": 1,
        "environmentVersion": 11,
        "observationVersion": 4,
        "observationSize": OBSERVATION_INPUT_SIZE,
        "plannerVersion": "test",
        "plannerDepth": 2,
        "seedPrefix": "base-test",
        "episodes": 1,
        "maxStepsPerEpisode": 2,
        "samples": 2,
        "actionOrder": [action.value for action in ACTION_ORDER],
        "actionCounts": {"forward": 2},
        "dataset": {
            "path": "dataset.npz",
            "sha256": digest,
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_dagger_collection_appends_student_visited_expert_labels(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    checkpoint = tmp_path / "student.pt"
    output = tmp_path / "dagger"
    _make_base(base)
    _make_student(checkpoint)

    manifest = collect_dagger_round(
        base_dataset=base,
        checkpoint=checkpoint,
        output=output,
        episodes=2,
        max_steps=2,
        depth=2,
        seed_prefix="dagger-test",
        device_name="cpu",
    )

    with np.load(output / "dataset.npz", allow_pickle=False) as data:
        assert len(data["observations"]) > 2
        assert len(data["actions"]) == len(data["observations"])
        assert len(data["student_actions"]) == len(data["observations"])
        assert len(data["source"]) == len(data["observations"])
        assert int(data["source"].sum()) > 0
        assert len(data["episode_offsets"]) == 4
        assert all(
            str(seed).startswith("dagger-test-aggregate:")
            for seed in data["seeds"]
        )

    assert manifest["aggregation"]["daggerEpisodes"] == 2
    assert manifest["aggregation"]["daggerSamples"] > 0
    assert 0.0 <= manifest["aggregation"]["disagreementRate"] <= 1.0
