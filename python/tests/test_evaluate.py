from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.connectome import load_default_reduced_graph
from fly_crossy.evaluate import (
    degree_preserving_rewire,
    evaluate,
    load_eval_config,
    summarize,
)
from fly_crossy.models import DensePolicy, FixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "eval-v1.json"


def test_summary_uses_distributions_and_terminal_breakdown() -> None:
    episodes = [
        {
            "score": 1.0,
            "survivalSteps": 4,
            "terminalReason": "vehicle",
            "actions": ["forward", "wait", "left", "forward"],
        },
        {
            "score": 5.0,
            "survivalSteps": 8,
            "terminalReason": "water",
            "actions": ["forward", "forward", "wait", "right"],
        },
        {
            "score": 3.0,
            "survivalSteps": 6,
            "terminalReason": "vehicle",
            "actions": ["forward", "backward", "left", "right"],
        },
    ]

    summary = summarize(episodes)

    assert summary["score"] == {"mean": 3.0, "median": 3.0, "max": 5.0}
    assert summary["survivalSteps"] == {
        "mean": 6.0,
        "median": 6.0,
        "max": 8.0,
    }
    assert summary["terminalReasons"] == {"vehicle": 2, "water": 1}
    assert sum(summary["terminalReasons"].values()) == len(episodes)
    assert summary["actionDistribution"]["forward"] == pytest.approx(5 / 12)
    assert summary["waitFrequency"] == pytest.approx(2 / 12)


def test_eval_seeds_do_not_overlap_training_seeds() -> None:
    config = load_eval_config(CONFIG)

    assert set(config.training_seeds).isdisjoint(config.evaluation_seeds)


def test_rewired_control_preserves_each_nodes_directed_degrees() -> None:
    graph = load_default_reduced_graph()

    rewired = degree_preserving_rewire(graph, seed="rewire-test", swaps=256)
    repeated = degree_preserving_rewire(graph, seed="rewire-test", swaps=256)

    original_out = np.bincount(graph.edge_index[0], minlength=graph.node_count)
    original_in = np.bincount(graph.edge_index[1], minlength=graph.node_count)
    rewired_out = np.bincount(rewired.edge_index[0], minlength=graph.node_count)
    rewired_in = np.bincount(rewired.edge_index[1], minlength=graph.node_count)
    assert np.array_equal(rewired_out, original_out)
    assert np.array_equal(rewired_in, original_in)
    assert len(set(map(tuple, rewired.edge_index.T.tolist()))) == graph.edge_count
    assert not np.array_equal(rewired.edge_index, graph.edge_index)
    assert np.array_equal(rewired.edge_index, repeated.edge_index)
    assert rewired.artifact_sha256 == repeated.artifact_sha256


def _write_checkpoint_pair(directory: Path) -> tuple[Path, Path]:
    torch.manual_seed(31)
    conventional = DensePolicy(
        observation_size=OBSERVATION_INPUT_SIZE,
        hidden_size=4,
        actions=len(ACTION_ORDER),
    )
    conventional_path = directory / "conventional.pt"
    torch.save(
        {
            "format_version": 1,
            "controller": "conventional",
            "model": {
                "observation_size": OBSERVATION_INPUT_SIZE,
                "hidden_size": 4,
                "actions": len(ACTION_ORDER),
            },
            "model_state_dict": conventional.state_dict(),
            "training": {"seed": "train-fixture", "steps": 8, "envs": 1},
        },
        conventional_path,
    )

    graph = load_default_reduced_graph()
    connectome = FixedGraphPolicy(
        graph, observation_size=OBSERVATION_INPUT_SIZE, actions=len(ACTION_ORDER)
    )
    connectome_path = directory / "connectome.pt"
    torch.save(
        {
            "format_version": 1,
            "controller": "connectome",
            "model": {
                "observation_size": OBSERVATION_INPUT_SIZE,
                "actions": len(ACTION_ORDER),
                "graph": graph.to_checkpoint(),
            },
            "model_state_dict": connectome.state_dict(),
            "training": {"seed": "train-fixture", "steps": 8, "envs": 1},
        },
        connectome_path,
    )
    return conventional_path, connectome_path


def test_evaluate_writes_versioned_json_csv_and_markdown_from_real_runs(
    tmp_path: Path,
) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": 1,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-a", "held-out-b"],
                "maxStepsPerEpisode": 4,
                "checkpoints": {
                    "conventional": conventional.name,
                    "connectome": connectome.name,
                },
                "controls": {
                    "rewiring": {"seed": "rewire-fixture", "swaps": 2},
                    "silencing": {"population": "sensory"},
                    "untrainedReadout": {"seed": 71},
                },
                "humanTraceFiles": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation"

    result = evaluate(config_path, output)
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    expected_ids = [
        "conventional",
        "reduced-connectome",
        "degree-preserving-rewired",
        "sensory-silenced",
        "untrained-readout",
    ]
    assert result["metricsJson"] == str(output / "metrics.json")
    assert result["metricsCsv"] == str(output / "metrics.csv")
    assert result["markdown"] == str(output / "summary.md")
    assert payload["version"] == 1
    assert payload["environmentVersion"] == 1
    assert payload["config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert payload["config"]["trainingSeeds"] == ["train-fixture"]
    assert payload["config"]["evaluationSeeds"] == ["held-out-a", "held-out-b"]
    assert payload["humanRecordedTraces"] == {
        "status": "not-present",
        "files": [],
        "episodeCount": 0,
    }
    assert [controller["id"] for controller in payload["controllers"]] == expected_ids
    assert [row["controllerId"] for row in rows] == expected_ids
    assert {row["version"] for row in rows} == {"1"}
    assert {row["environmentVersion"] for row in rows} == {"1"}
    assert {row["configSha256"] for row in rows} == {payload["config"]["sha256"]}
    for controller in payload["controllers"]:
        assert [episode["seed"] for episode in controller["episodes"]] == [
            "held-out-a",
            "held-out-b",
        ]
        assert sum(controller["summary"]["terminalReasons"].values()) == 2
        assert controller["evidence"]["modelParameters"] > 0
        assert controller["evidence"]["trainingEnvironmentSteps"] == 8
        assert len(controller["evidence"]["checkpointSha256"]) == 64
        assert controller["summary"]["inferencePerformance"]["calls"] > 0
        assert controller["summary"]["inferencePerformance"]["wallSeconds"] >= 0
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "No human-recorded traces were configured." in markdown
    assert "## Action distributions" in markdown
    assert "## Inference performance" in markdown
    assert "## Evidence hashes" in markdown
    assert "Terminal reasons" in markdown
    assert hashlib.sha256(conventional.read_bytes()).hexdigest() in markdown


def test_evaluate_replays_configured_human_actions_without_inventing_activity(
    tmp_path: Path,
) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    trace_path = tmp_path / "human-trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": 1,
                "source": {"kind": "human-recorded", "name": "test participant"},
                "episodes": [
                    {
                        "seed": "held-out-human",
                        "actions": ["forward", "wait", "left"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "eval-human.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": 1,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-human"],
                "maxStepsPerEpisode": 3,
                "checkpoints": {
                    "conventional": conventional.name,
                    "connectome": connectome.name,
                },
                "controls": {
                    "rewiring": {"seed": "rewire-fixture", "swaps": 2},
                    "silencing": {"population": "sensory"},
                    "untrainedReadout": {"seed": 71},
                },
                "humanTraceFiles": [trace_path.name],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evaluation-human"

    evaluate(config_path, output)
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    human = payload["humanRecordedTraces"]

    assert human["status"] == "evaluated"
    assert human["episodeCount"] == 1
    assert human["files"] == [
        {
            "path": str(trace_path),
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        }
    ]
    assert human["episodes"][0]["seed"] == "held-out-human"
    assert human["episodes"][0]["actions"] == ["forward", "wait", "left"]
    assert "activity" not in human["episodes"][0]
    assert human["summary"]["episodeCount"] == 1
    assert payload["controllers"][0]["id"] == "human-recorded"
