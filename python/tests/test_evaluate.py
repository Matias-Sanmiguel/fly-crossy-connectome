from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fly_crossy.connectome import load_default_reduced_graph
from fly_crossy.env import WORLD_VERSION
from fly_crossy.evaluate import (
    _load_checkpoint,
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


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("version",), True),
        (("version",), "1"),
        (("environmentVersion",), 1.0),
        (("maxStepsPerEpisode",), 4.5),
        (("controls", "rewiring", "swaps"), False),
        (("controls", "untrainedReadout", "seed"), "71"),
    ],
)
def test_eval_config_rejects_coercible_integer_fields(
    tmp_path: Path, path: tuple[str, ...], invalid: object
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = invalid
    config_path = tmp_path / "invalid-eval.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="integer"):
        load_eval_config(config_path)


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


def test_rewired_control_renormalizes_absolute_incoming_weight_per_target() -> None:
    graph = load_default_reduced_graph()

    rewired = degree_preserving_rewire(graph, seed="rewire-normalization", swaps=256)

    original_sums = np.bincount(
        graph.edge_index[1], weights=np.abs(graph.edge_weight), minlength=graph.node_count
    )
    rewired_sums = np.bincount(
        rewired.edge_index[1],
        weights=np.abs(rewired.edge_weight),
        minlength=rewired.node_count,
    )
    connected_targets = np.bincount(
        graph.edge_index[1], minlength=graph.node_count
    ) > 0
    assert rewired_sums[connected_targets] == pytest.approx(
        original_sums[connected_targets], abs=1e-6
    )


def _write_checkpoint_pair(
    directory: Path,
    *,
    conventional_environment_version: int | None = WORLD_VERSION,
    connectome_environment_version: int | None = WORLD_VERSION,
) -> tuple[Path, Path]:
    torch.manual_seed(31)
    conventional = DensePolicy(
        observation_size=OBSERVATION_INPUT_SIZE,
        hidden_size=4,
        actions=len(ACTION_ORDER),
    )
    conventional_path = directory / "conventional.pt"
    conventional_checkpoint = {
        "format_version": 1,
        "controller": "conventional",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "hidden_size": 4,
            "actions": len(ACTION_ORDER),
        },
        "model_state_dict": conventional.state_dict(),
        "training": {
            "seed": "train-fixture",
            "steps": 8,
            "envs": 1,
            "learning_rate": 0.0003,
            "world_seeds": ["train-fixture:0:0"],
        },
    }
    if conventional_environment_version is not None:
        conventional_checkpoint["environment_version"] = conventional_environment_version
    torch.save(conventional_checkpoint, conventional_path)

    graph = load_default_reduced_graph()
    connectome = FixedGraphPolicy(
        graph, observation_size=OBSERVATION_INPUT_SIZE, actions=len(ACTION_ORDER)
    )
    connectome_path = directory / "connectome.pt"
    connectome_checkpoint = {
        "format_version": 1,
        "controller": "connectome",
        "model": {
            "observation_size": OBSERVATION_INPUT_SIZE,
            "actions": len(ACTION_ORDER),
            "graph": graph.to_checkpoint(),
        },
        "model_state_dict": connectome.state_dict(),
        "training": {
            "seed": "train-fixture",
            "steps": 8,
            "envs": 1,
            "learning_rate": 0.0003,
            "world_seeds": ["train-fixture:0:0"],
        },
    }
    if connectome_environment_version is not None:
        connectome_checkpoint["environment_version"] = connectome_environment_version
    torch.save(connectome_checkpoint, connectome_path)
    return conventional_path, connectome_path


def _checkpoint_reference(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    ("field_path", "invalid"),
    [
        pytest.param(("format_version",), True, id="boolean-format-version"),
        pytest.param(("model", "observation_size"), "370", id="string-observation-size"),
        pytest.param(("model", "observation_size"), 370.9, id="fractional-observation-size"),
        pytest.param(("model", "actions"), 5.9, id="fractional-actions"),
        pytest.param(("model", "hidden_size"), 64.9, id="fractional-hidden-size"),
        pytest.param(("model",), [], id="model-not-mapping"),
        pytest.param(("model_state_dict",), [], id="state-dict-not-mapping"),
        pytest.param(("training",), [], id="training-not-mapping"),
        pytest.param(("training", "learning_rate"), True, id="boolean-learning-rate"),
        pytest.param(("environment_version",), None, id="missing-environment-version"),
        pytest.param(("environment_version",), WORLD_VERSION + 1, id="unsupported-environment-version"),
    ],
)
def test_evaluator_rejects_untrusted_checkpoint_metadata_before_conversion(
    tmp_path: Path, field_path: tuple[str, ...], invalid: object
) -> None:
    conventional, _ = _write_checkpoint_pair(tmp_path)
    payload = torch.load(conventional, map_location="cpu", weights_only=True)
    if invalid is None:
        del payload[field_path[0]]
    else:
        cursor = payload
        for key in field_path[:-1]:
            cursor = cursor[key]
        cursor[field_path[-1]] = invalid
    torch.save(payload, conventional)

    with pytest.raises(ValueError, match="[Cc]heckpoint"):
        _load_checkpoint(
            conventional,
            hashlib.sha256(conventional.read_bytes()).hexdigest(),
            "conventional",
            ["train-fixture"],
            WORLD_VERSION,
        )


def test_evaluator_rejects_non_mapping_checkpoint_and_nonfinite_tensor(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save([], checkpoint)
    with pytest.raises(ValueError, match="checkpoint must contain an object"):
        _load_checkpoint(
            checkpoint,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "conventional",
            ["train-fixture"],
            WORLD_VERSION,
        )

    checkpoint, _ = _write_checkpoint_pair(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["actor.bias"][0] = float("inf")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="finite"):
        _load_checkpoint(
            checkpoint,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "conventional",
            ["train-fixture"],
            WORLD_VERSION,
        )

    checkpoint, _ = _write_checkpoint_pair(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model_state_dict"]["actor.bias"] = torch.zeros(5, dtype=torch.bool)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="floating-point"):
        _load_checkpoint(
            checkpoint,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "conventional",
            ["train-fixture"],
            WORLD_VERSION,
        )


def test_evaluate_writes_versioned_json_csv_and_markdown_from_real_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": WORLD_VERSION,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-a", "held-out-b"],
                "maxStepsPerEpisode": 4,
                "checkpoints": {
                    "conventional": _checkpoint_reference(conventional),
                    "connectome": _checkpoint_reference(connectome),
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

    monkeypatch.chdir(tmp_path)
    result = evaluate(Path("eval.json"), output)
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert b"\r\n" not in (output / "metrics.csv").read_bytes()

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
    assert payload["environmentVersion"] == WORLD_VERSION
    assert payload["config"]["path"] == "eval.json"
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
    assert {row["environmentVersion"] for row in rows} == {str(WORLD_VERSION)}
    assert {row["configSha256"] for row in rows} == {payload["config"]["sha256"]}
    assert {row["checkpointEnvironmentVersion"] for row in rows} == {str(WORLD_VERSION)}
    assert {row["checkpointEnvironmentVersionStatus"] for row in rows} == {
        "recorded-match"
    }
    for controller in payload["controllers"]:
        assert [episode["seed"] for episode in controller["episodes"]] == [
            "held-out-a",
            "held-out-b",
        ]
        assert sum(controller["summary"]["terminalReasons"].values()) == 2
        assert controller["evidence"]["modelParameters"] > 0
        assert controller["evidence"]["trainingEnvironmentSteps"] == 8
        assert len(controller["evidence"]["checkpointSha256"]) == 64
        assert controller["evidence"]["checkpointEnvironmentVersion"] == WORLD_VERSION
        assert controller["evidence"]["trainingWorldSeeds"] == [
            "train-fixture:0:0"
        ]
        assert (
            controller["evidence"]["checkpointEnvironmentVersionStatus"]
            == "recorded-match"
        )
        assert controller["summary"]["inferencePerformance"]["calls"] > 0
        assert controller["summary"]["inferencePerformance"]["wallSeconds"] >= 0
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "No human-recorded traces were configured." in markdown
    assert "## Action distributions" in markdown
    assert "## Inference performance" in markdown
    assert "## Evidence hashes" in markdown
    assert "Terminal reasons" in markdown
    assert "does not isolate topology" in markdown
    assert hashlib.sha256(conventional.read_bytes()).hexdigest() in markdown
    assert "recorded-match" in markdown


def test_evaluate_rejects_checkpoint_environment_version_mismatch(
    tmp_path: Path,
) -> None:
    conventional, connectome = _write_checkpoint_pair(
        tmp_path,
        conventional_environment_version=WORLD_VERSION + 1,
        connectome_environment_version=WORLD_VERSION,
    )
    config_path = tmp_path / "eval-mismatch.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": WORLD_VERSION,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-a"],
                "maxStepsPerEpisode": 4,
                "checkpoints": {
                    "conventional": _checkpoint_reference(conventional),
                    "connectome": _checkpoint_reference(connectome),
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

    with pytest.raises(
        ValueError,
        match=(
            f"Checkpoint environment version {WORLD_VERSION + 1} does not match "
            f"evaluation environment version {WORLD_VERSION}"
        ),
    ):
        evaluate(config_path, tmp_path / "evaluation-mismatch")


def test_evaluate_replays_configured_human_actions_without_inventing_activity(
    tmp_path: Path,
) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    trace_path = tmp_path / "human-trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": WORLD_VERSION,
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
                "environmentVersion": WORLD_VERSION,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-human"],
                "maxStepsPerEpisode": 3,
                "checkpoints": {
                    "conventional": _checkpoint_reference(conventional),
                    "connectome": _checkpoint_reference(connectome),
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


def test_evaluate_verifies_expected_checkpoint_hash_before_loading(tmp_path: Path) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    config_path = tmp_path / "eval-hash.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": WORLD_VERSION,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["held-out-a"],
                "maxStepsPerEpisode": 4,
                "checkpoints": {
                    "conventional": {
                        "path": conventional.name,
                        "sha256": "0" * 64,
                    },
                    "connectome": _checkpoint_reference(connectome),
                },
                "controls": {
                    "rewiring": {"seed": "rewire-fixture", "swaps": 2},
                    "silencing": {"population": "sensory"},
                    "untrainedReadout": {"seed": 71},
                },
                "humanTraceFiles": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        evaluate(config_path, tmp_path / "evaluation")


def test_evaluate_rejects_concrete_training_world_seed_overlap(tmp_path: Path) -> None:
    conventional, connectome = _write_checkpoint_pair(tmp_path)
    for checkpoint_path in (conventional, connectome):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["training"]["world_seeds"] = ["train-fixture:0:0"]
        torch.save(checkpoint, checkpoint_path)
    config_path = tmp_path / "eval-leak.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "environmentVersion": WORLD_VERSION,
                "trainingSeeds": ["train-fixture"],
                "evaluationSeeds": ["train-fixture:0:0"],
                "maxStepsPerEpisode": 4,
                "checkpoints": {
                    "conventional": _checkpoint_reference(conventional),
                    "connectome": _checkpoint_reference(connectome),
                },
                "controls": {
                    "rewiring": {"seed": "rewire-fixture", "swaps": 2},
                    "silencing": {"population": "sensory"},
                    "untrainedReadout": {"seed": 71},
                },
                "humanTraceFiles": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="concrete training world seeds overlap"):
        evaluate(config_path, tmp_path / "evaluation")
