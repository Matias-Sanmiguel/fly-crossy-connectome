from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .connectome import ReducedGraphArtifact
from .checkpoint import validate_checkpoint
from .env import FlyCrossyEnv, WORLD_VERSION, hash_seed
from .models import (
    DensePolicy,
    FeedbackNestedPopulationFixedGraphPolicy,
    FixedGraphPolicy,
    GatedNestedPopulationFixedGraphPolicy,
    NestedPopulationFixedGraphPolicy,
    PopulationFixedGraphPolicy,
)
from .schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE, Action


@dataclass(frozen=True, slots=True)
class EvalConfig:
    version: int
    environment_version: int
    training_seeds: tuple[str, ...]
    evaluation_seeds: tuple[str, ...]
    max_steps_per_episode: int
    conventional_checkpoint: Path
    conventional_checkpoint_sha256: str
    connectome_checkpoint: Path
    connectome_checkpoint_sha256: str
    rewiring_seed: str
    rewiring_swaps: int
    silencing_population: str
    untrained_readout_seed: int
    human_trace_files: tuple[Path, ...]


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Evaluation {label} must be a non-empty string.")
    return value.strip()


def _required_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Evaluation {label} must be an integer.")
    return value


def _checkpoint_reference(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation {label} checkpoint must be an object.")
    path = _required_string(value.get("path"), f"{label} checkpoint path")
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"Evaluation {label} checkpoint SHA-256 must be a lowercase digest."
        )
    return path, digest


def _string_tuple(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"Evaluation {label} must be a non-empty list.")
    result = tuple(_required_string(item, f"{label} entry") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"Evaluation {label} must not contain duplicates.")
    return result


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load and validate the versioned held-out evaluation configuration."""
    config_path = Path(path).resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluation config must be valid JSON.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation config must be a JSON object.")
    try:
        version = _required_integer(payload["version"], "version")
        environment_version = _required_integer(
            payload["environmentVersion"], "environment version"
        )
        training_seeds = _string_tuple(payload["trainingSeeds"], "training seeds")
        evaluation_seeds = _string_tuple(payload["evaluationSeeds"], "evaluation seeds")
        max_steps = _required_integer(
            payload["maxStepsPerEpisode"], "max steps per episode"
        )
        checkpoints = payload["checkpoints"]
        controls = payload["controls"]
        rewiring = controls["rewiring"]
        silencing = controls["silencing"]
        untrained = controls["untrainedReadout"]
        human_files = _string_tuple(
            payload.get("humanTraceFiles", []), "human trace files", allow_empty=True
        )
        conventional, conventional_sha256 = _checkpoint_reference(
            checkpoints["conventional"], "conventional"
        )
        connectome, connectome_sha256 = _checkpoint_reference(
            checkpoints["connectome"], "connectome"
        )
        rewiring_seed = _required_string(rewiring["seed"], "rewiring seed")
        rewiring_swaps = _required_integer(rewiring["swaps"], "rewiring swaps")
        silencing_population = _required_string(
            silencing["population"], "silencing population"
        )
        untrained_seed = _required_integer(
            untrained["seed"], "untrained readout seed"
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Evaluation "):
            raise
        raise ValueError("Evaluation config is missing a compatible required field.") from error

    if version != 1:
        raise ValueError("Evaluation config version must be 1.")
    if environment_version != WORLD_VERSION:
        raise ValueError(
            f"Evaluation environment version must be {WORLD_VERSION}."
        )
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("Evaluation seeds must not overlap training seeds.")
    if max_steps <= 0:
        raise ValueError("Evaluation max steps per episode must be positive.")
    if rewiring_swaps <= 0:
        raise ValueError("Evaluation rewiring swaps must be positive.")
    if silencing_population not in ("sensory", "readout"):
        raise ValueError("Evaluation silencing population must be sensory or readout.")

    base = config_path.parent
    return EvalConfig(
        version=version,
        environment_version=environment_version,
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
        max_steps_per_episode=max_steps,
        conventional_checkpoint=(base / conventional).resolve(),
        conventional_checkpoint_sha256=conventional_sha256,
        connectome_checkpoint=(base / connectome).resolve(),
        connectome_checkpoint_sha256=connectome_sha256,
        rewiring_seed=rewiring_seed,
        rewiring_swaps=rewiring_swaps,
        silencing_population=silencing_population,
        untrained_readout_seed=untrained_seed,
        human_trace_files=tuple((base / value).resolve() for value in human_files),
    )


def _distribution(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty episode collection.")
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize complete episodes with distributions and exhaustive outcomes."""
    if not episodes:
        raise ValueError("Cannot summarize an empty episode collection.")
    scores = [float(episode["score"]) for episode in episodes]
    survival_steps = [int(episode["survivalSteps"]) for episode in episodes]
    terminal_reasons = Counter(str(episode["terminalReason"]) for episode in episodes)
    actions = Counter(
        str(action) for episode in episodes for action in episode.get("actions", [])
    )
    total_actions = sum(actions.values())
    action_counts = {action.value: actions[action.value] for action in ACTION_ORDER}
    action_distribution = {
        action.value: (
            float(actions[action.value] / total_actions) if total_actions else 0.0
        )
        for action in ACTION_ORDER
    }
    return {
        "episodeCount": len(episodes),
        "score": _distribution(scores),
        "survivalSteps": _distribution(survival_steps),
        "terminalReasons": dict(sorted(terminal_reasons.items())),
        "actionCounts": action_counts,
        "actionDistribution": action_distribution,
        "waitFrequency": action_distribution["wait"],
    }


def degree_preserving_rewire(
    graph: ReducedGraphArtifact, *, seed: str, swaps: int
) -> ReducedGraphArtifact:
    """Return a deterministic directed double-edge-swap control graph."""
    if not seed:
        raise ValueError("Rewiring seed must not be empty.")
    if swaps <= 0:
        raise ValueError("Rewiring swaps must be positive.")
    edge_index = graph.edge_index.copy()
    edges = {tuple(int(value) for value in edge) for edge in edge_index.T}
    rng = np.random.default_rng(hash_seed(seed))
    completed = 0
    attempts = 0
    maximum_attempts = max(100, swaps * 50)
    while completed < swaps and attempts < maximum_attempts:
        attempts += 1
        first, second = rng.choice(graph.edge_count, size=2, replace=False).tolist()
        source_a, target_a = (int(value) for value in edge_index[:, first])
        source_b, target_b = (int(value) for value in edge_index[:, second])
        if source_a == source_b or target_a == target_b:
            continue
        candidate_a = (source_a, target_b)
        candidate_b = (source_b, target_a)
        if source_a == target_b or source_b == target_a:
            continue
        old_a = (source_a, target_a)
        old_b = (source_b, target_b)
        edges.remove(old_a)
        edges.remove(old_b)
        if candidate_a in edges or candidate_b in edges:
            edges.add(old_a)
            edges.add(old_b)
            continue
        edge_index[1, first] = target_b
        edge_index[1, second] = target_a
        edges.add(candidate_a)
        edges.add(candidate_b)
        completed += 1

    if completed != swaps:
        raise ValueError(
            f"Could not complete {swaps} degree-preserving swaps after {attempts} attempts."
        )
    original_incoming = np.bincount(
        graph.edge_index[1],
        weights=np.abs(graph.edge_weight),
        minlength=graph.node_count,
    )
    edge_weight = graph.edge_weight.copy()
    rewired_incoming = np.bincount(
        edge_index[1], weights=np.abs(edge_weight), minlength=graph.node_count
    )
    connected_targets = np.bincount(
        edge_index[1], minlength=graph.node_count
    ) > 0
    if np.any(original_incoming[connected_targets] <= 0) or np.any(
        rewired_incoming[connected_targets] <= 0
    ):
        raise ValueError("Connected targets require non-zero absolute incoming weight.")
    target_scales = np.ones(graph.node_count, dtype=np.float64)
    target_scales[connected_targets] = (
        original_incoming[connected_targets] / rewired_incoming[connected_targets]
    )
    edge_weight = np.asarray(
        edge_weight * target_scales[edge_index[1]], dtype=np.float32
    )
    return ReducedGraphArtifact(
        dataset_version=graph.dataset_version,
        source_url=graph.source_url,
        license=graph.license,
        source_sha256=graph.source_sha256,
        selection_rule=(
            f"{graph.selection_rule} Degree-preserving evaluator control: "
            f"{swaps} directed target swaps with seed {seed}; signed weights "
            "renormalized to preserve each target's absolute incoming sum."
        ),
        minimum_edge_threshold=graph.minimum_edge_threshold,
        body_ids=graph.body_ids.copy(),
        edge_index=edge_index,
        edge_weight=edge_weight,
        sensory_body_ids=graph.sensory_body_ids.copy(),
        readout_body_ids=graph.readout_body_ids.copy(),
    )


@dataclass(slots=True)
class _PolicyRunner:
    model: DensePolicy | FixedGraphPolicy
    silenced_indices: tuple[int, ...] = ()
    hidden: Tensor | None = None

    def reset(self) -> None:
        self.hidden = (
            torch.zeros(1, self.model.graph.node_count, dtype=torch.float32)
            if isinstance(self.model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy))
            else None
        )

    def decide(self, observation: np.ndarray[Any, np.dtype[np.float32]]) -> tuple[Action, float]:
        inputs = torch.from_numpy(observation).unsqueeze(0)
        started = perf_counter()
        with torch.no_grad():
            if isinstance(self.model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)):
                if self.hidden is None:
                    raise RuntimeError("Reset the fixed graph evaluator before inference.")
                logits, _, activity = self.model(inputs, self.hidden)
                if self.silenced_indices:
                    activity = activity.clone()
                    activity[:, self.silenced_indices] = 0
                    logits = self.model.actor_from_activity(activity)
                self.hidden = activity
            else:
                logits, _ = self.model(inputs)
            action_index = int(torch.argmax(logits, dim=-1).item())
        elapsed = perf_counter() - started
        return ACTION_ORDER[action_index], elapsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_controller: str,
    training_seeds: Sequence[str],
    expected_environment_version: int,
) -> tuple[DensePolicy | FixedGraphPolicy, dict[str, Any], int | None]:
    if not path.is_file():
        raise ValueError(f"Evaluation checkpoint does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Evaluation checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Evaluation checkpoint could not be loaded: {path}") from error
    validated = validate_checkpoint(
        checkpoint,
        expected_controller=expected_controller,
        expected_environment_version=expected_environment_version,
        expected_observation_size=OBSERVATION_INPUT_SIZE,
    )
    state_dict = validated.state_dict
    training = validated.training
    observation_size = validated.observation_size
    actions = validated.actions
    training_seed = training["seed"]
    checkpoint_environment_version = validated.environment_version
    if training_seed not in training_seeds:
        raise ValueError(
            f"Checkpoint training seed {training_seed!r} is not declared in trainingSeeds."
        )
    if expected_controller == "conventional":
        hidden_size = validated.hidden_size
        if hidden_size is None:
            raise ValueError("Conventional checkpoint hidden size is incompatible.")
        model: DensePolicy | FixedGraphPolicy = DensePolicy(
            observation_size, hidden_size, actions
        )
    else:
        graph = validated.graph
        if graph is None:
            raise ValueError("Connectome checkpoint graph is incompatible.")
        if validated.connectome_interface == "nested-feedback":
            core_graph = validated.core_graph
            if core_graph is None:
                raise ValueError("Connectome nested core graph is incompatible.")
            model = FeedbackNestedPopulationFixedGraphPolicy(
                graph, core_graph, observation_size, actions
            )
        elif validated.connectome_interface == "nested-gated":
            core_graph = validated.core_graph
            if core_graph is None:
                raise ValueError("Connectome nested core graph is incompatible.")
            model = GatedNestedPopulationFixedGraphPolicy(
                graph, core_graph, observation_size, actions
            )
        elif validated.connectome_interface == "nested":
            core_graph = validated.core_graph
            if core_graph is None:
                raise ValueError("Connectome nested core graph is incompatible.")
            model = NestedPopulationFixedGraphPolicy(
                graph, core_graph, observation_size, actions
            )
        elif validated.connectome_interface == "population":
            model = PopulationFixedGraphPolicy(
                graph, observation_size, actions
            )
        else:
            model = FixedGraphPolicy(graph, observation_size, actions)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Evaluation checkpoint parameters are incompatible.") from error
    model.eval()
    return model, training, checkpoint_environment_version


def _parameter_count(model: DensePolicy | FixedGraphPolicy) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


def _evaluate_runner(
    runner: _PolicyRunner, seeds: Sequence[str], maximum_steps: int
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        environment = FlyCrossyEnv()
        observation, info = environment.reset(seed)
        runner.reset()
        actions: list[str] = []
        total_return = 0.0
        inference_seconds = 0.0
        terminal_reason: str | None = None
        for _ in range(maximum_steps):
            action, elapsed = runner.decide(observation)
            inference_seconds += elapsed
            actions.append(action.value)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += reward
            if terminated or truncated:
                terminal_reason = str(info["terminalReason"])
                break
        if terminal_reason is None:
            terminal_reason = "step-limit"
        episodes.append(
            {
                "seed": seed,
                "score": float(info["score"]),
                "return": float(total_return),
                "survivalSteps": len(actions),
                "terminalReason": terminal_reason,
                "actions": actions,
                "inferenceWallSeconds": float(inference_seconds),
            }
        )
    return episodes


def _evidence(
    model: DensePolicy | FixedGraphPolicy,
    checkpoint_path: Path,
    training: Mapping[str, Any],
    *,
    checkpoint_environment_version: int | None,
    control_graph_hash: str | None = None,
    silenced_body_ids: Sequence[int] = (),
) -> dict[str, Any]:
    parameters, trainable_parameters = _parameter_count(model)
    graph_hash = (
        model.graph.artifact_sha256 if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)) else None
    )
    return {
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": _sha256(checkpoint_path),
        "checkpointController": (
            "connectome" if isinstance(model, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy, FeedbackNestedPopulationFixedGraphPolicy)) else "conventional"
        ),
        "trainingSeed": str(training["seed"]),
        "trainingEnvironmentSteps": int(training["steps"]),
        "trainingWorldSeeds": list(training["world_seeds"]),
        "checkpointEnvironmentVersion": checkpoint_environment_version,
        "checkpointEnvironmentVersionStatus": (
            "recorded-match"
            if checkpoint_environment_version is not None
            else "legacy-unrecorded"
        ),
        "modelParameters": parameters,
        "trainableParameters": trainable_parameters,
        "graphArtifactSha256": graph_hash,
        "controlGraphArtifactSha256": control_graph_hash,
        "silencedBodyIds": [int(body_id) for body_id in silenced_body_ids],
    }


def _controller_result(
    *,
    controller_id: str,
    label: str,
    control: str,
    runner: _PolicyRunner,
    seeds: Sequence[str],
    maximum_steps: int,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    episodes = _evaluate_runner(runner, seeds, maximum_steps)
    summary = summarize(episodes)
    calls = sum(int(episode["survivalSteps"]) for episode in episodes)
    wall_seconds = sum(float(episode["inferenceWallSeconds"]) for episode in episodes)
    summary["inferencePerformance"] = {
        "calls": calls,
        "wallSeconds": wall_seconds,
        "meanMilliseconds": (wall_seconds * 1000 / calls if calls else 0.0),
        "stepsPerSecond": (calls / wall_seconds if wall_seconds else 0.0),
    }
    return {
        "id": controller_id,
        "label": label,
        "control": control,
        "evidence": evidence,
        "episodes": episodes,
        "summary": summary,
    }


def _silenced_indices(graph: ReducedGraphArtifact, population: str) -> tuple[int, ...]:
    body_ids = (
        graph.sensory_body_ids if population == "sensory" else graph.readout_body_ids
    )
    index_by_body_id = {
        int(body_id): index for index, body_id in enumerate(graph.body_ids)
    }
    return tuple(index_by_body_id[int(body_id)] for body_id in body_ids)


def _evaluate_human_traces(
    paths: Sequence[Path], evaluation_seeds: Sequence[str], maximum_steps: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []
    allowed_seeds = set(evaluation_seeds)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Human trace file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Human trace file must be valid JSON: {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("Human trace file must contain an object.")
        try:
            if payload["version"] != 1 or payload["environmentVersion"] != WORLD_VERSION:
                raise ValueError("Human trace version or environment version is incompatible.")
            source = payload["source"]
            if source["kind"] != "human-recorded":
                raise ValueError("Human trace source kind must be human-recorded.")
            trace_episodes = payload["episodes"]
        except (KeyError, TypeError) as error:
            raise ValueError("Human trace metadata is incompatible.") from error
        if not isinstance(trace_episodes, list) or not trace_episodes:
            raise ValueError("Human trace file must contain at least one episode.")
        files.append({"path": str(path), "sha256": _sha256(path)})
        for trace in trace_episodes:
            if not isinstance(trace, Mapping):
                raise ValueError("Human trace episode must be an object.")
            seed = _required_string(trace.get("seed"), "human trace seed")
            if seed not in allowed_seeds:
                raise ValueError(
                    f"Human trace seed {seed!r} is not in the held-out evaluation suite."
                )
            action_values = trace.get("actions")
            if not isinstance(action_values, list) or not action_values:
                raise ValueError("Human trace episode actions must be a non-empty list.")
            try:
                trace_actions = [Action(value) for value in action_values]
            except (TypeError, ValueError) as error:
                raise ValueError("Human trace contains an unknown action.") from error

            environment = FlyCrossyEnv()
            _, info = environment.reset(seed)
            replayed: list[str] = []
            total_return = 0.0
            terminal_reason: str | None = None
            for action in trace_actions[:maximum_steps]:
                _, reward, terminated, truncated, info = environment.step(action)
                replayed.append(action.value)
                total_return += reward
                if terminated or truncated:
                    terminal_reason = str(info["terminalReason"])
                    break
            if terminal_reason is None:
                terminal_reason = (
                    "step-limit"
                    if len(replayed) == maximum_steps
                    else "trace-ended"
                )
            episodes.append(
                {
                    "seed": seed,
                    "score": float(info["score"]),
                    "return": float(total_return),
                    "survivalSteps": len(replayed),
                    "terminalReason": terminal_reason,
                    "actions": replayed,
                }
            )

    summary = summarize(episodes)
    summary["inferencePerformance"] = {
        "calls": 0,
        "wallSeconds": 0.0,
        "meanMilliseconds": 0.0,
        "stepsPerSecond": 0.0,
    }
    human_recorded = {
        "status": "evaluated",
        "files": files,
        "episodeCount": len(episodes),
        "episodes": episodes,
        "summary": summary,
    }
    controller = {
        "id": "human-recorded",
        "label": "Human-recorded traces",
        "control": "Recorded action trace replay",
        "evidence": {
            "checkpointPath": None,
            "checkpointSha256": None,
            "checkpointController": "human-recorded",
            "trainingSeed": None,
            "trainingEnvironmentSteps": 0,
            "checkpointEnvironmentVersion": None,
            "checkpointEnvironmentVersionStatus": "not-applicable",
            "modelParameters": 0,
            "trainableParameters": 0,
            "graphArtifactSha256": None,
            "controlGraphArtifactSha256": None,
            "silencedBodyIds": [],
            "traceFileSha256": [item["sha256"] for item in files],
        },
        "episodes": episodes,
        "summary": summary,
    }
    return human_recorded, controller


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fieldnames = [
        "version",
        "environmentVersion",
        "configSha256",
        "controllerId",
        "label",
        "control",
        "episodeCount",
        "scoreMean",
        "scoreMedian",
        "scoreMax",
        "survivalStepsMean",
        "survivalStepsMedian",
        "survivalStepsMax",
        "terminalReasons",
        "actionDistribution",
        "waitFrequency",
        "modelParameters",
        "trainableParameters",
        "trainingEnvironmentSteps",
        "checkpointEnvironmentVersion",
        "checkpointEnvironmentVersionStatus",
        "inferenceCalls",
        "inferenceWallSeconds",
        "inferenceMeanMilliseconds",
        "inferenceStepsPerSecond",
        "checkpointSha256",
        "graphArtifactSha256",
        "controlGraphArtifactSha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for controller in payload["controllers"]:
            summary = controller["summary"]
            evidence = controller["evidence"]
            inference = summary["inferencePerformance"]
            writer.writerow(
                {
                    "version": payload["version"],
                    "environmentVersion": payload["environmentVersion"],
                    "configSha256": payload["config"]["sha256"],
                    "controllerId": controller["id"],
                    "label": controller["label"],
                    "control": controller["control"],
                    "episodeCount": summary["episodeCount"],
                    "scoreMean": summary["score"]["mean"],
                    "scoreMedian": summary["score"]["median"],
                    "scoreMax": summary["score"]["max"],
                    "survivalStepsMean": summary["survivalSteps"]["mean"],
                    "survivalStepsMedian": summary["survivalSteps"]["median"],
                    "survivalStepsMax": summary["survivalSteps"]["max"],
                    "terminalReasons": json.dumps(
                        summary["terminalReasons"], sort_keys=True, separators=(",", ":")
                    ),
                    "actionDistribution": json.dumps(
                        summary["actionDistribution"], sort_keys=True, separators=(",", ":")
                    ),
                    "waitFrequency": summary["waitFrequency"],
                    "modelParameters": evidence["modelParameters"],
                    "trainableParameters": evidence["trainableParameters"],
                    "trainingEnvironmentSteps": evidence["trainingEnvironmentSteps"],
                    "checkpointEnvironmentVersion": (
                        evidence["checkpointEnvironmentVersion"]
                        if evidence["checkpointEnvironmentVersion"] is not None
                        else ""
                    ),
                    "checkpointEnvironmentVersionStatus": evidence[
                        "checkpointEnvironmentVersionStatus"
                    ],
                    "inferenceCalls": inference["calls"],
                    "inferenceWallSeconds": inference["wallSeconds"],
                    "inferenceMeanMilliseconds": inference["meanMilliseconds"],
                    "inferenceStepsPerSecond": inference["stepsPerSecond"],
                    "checkpointSha256": evidence["checkpointSha256"],
                    "graphArtifactSha256": evidence["graphArtifactSha256"] or "",
                    "controlGraphArtifactSha256": evidence["controlGraphArtifactSha256"] or "",
                }
            )


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Evaluation v1 emitted summary",
        "",
        (
            f"Environment v{payload['environmentVersion']}; "
            f"{len(payload['config']['evaluationSeeds'])} held-out seeds; "
            f"maximum {payload['config']['maxStepsPerEpisode']} steps per episode."
        ),
        "",
    ]
    if payload["humanRecordedTraces"]["status"] == "not-present":
        lines.extend(
            [
                "No human-recorded traces were configured. No human baseline was fabricated.",
                "",
            ]
        )
    lines.extend(
        [
            "| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for controller in payload["controllers"]:
        summary = controller["summary"]
        evidence = controller["evidence"]
        score = summary["score"]
        survival = summary["survivalSteps"]
        terminal_reasons = ", ".join(
            f"{reason} {count}"
            for reason, count in summary["terminalReasons"].items()
        )
        lines.append(
            "| "
            + str(controller["label"])
            + " | "
            + " / ".join(_format_number(score[key]) for key in ("mean", "median", "max"))
            + " | "
            + " / ".join(
                _format_number(survival[key]) for key in ("mean", "median", "max")
            )
            + f" | {_format_number(summary['waitFrequency'])}"
            + f" | {terminal_reasons}"
            + f" | {evidence['trainingEnvironmentSteps']}"
            + f" | {evidence['modelParameters']} |"
        )
    lines.extend(
        [
            "",
            "## Action distributions",
            "",
            "Each action cell is `count / frequency`.",
            "",
            "| Controller / control | Forward | Backward | Left | Right | Wait |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller in payload["controllers"]:
        summary = controller["summary"]
        action_cells = [
            (
                f"{summary['actionCounts'][action.value]} / "
                f"{_format_number(summary['actionDistribution'][action.value])}"
            )
            for action in ACTION_ORDER
        ]
        lines.append(
            f"| {controller['label']} | " + " | ".join(action_cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Inference performance",
            "",
            "Wall-clock values include model decision calls only and are machine-dependent.",
            "",
            "| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller in payload["controllers"]:
        inference = controller["summary"]["inferencePerformance"]
        lines.append(
            f"| {controller['label']} | {inference['calls']}"
            f" | {inference['wallSeconds']:.6f}"
            f" | {inference['meanMilliseconds']:.6f}"
            f" | {inference['stepsPerSecond']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence hashes",
            "",
            f"Config SHA-256: `{payload['config']['sha256']}`.",
            "",
            "| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for controller in payload["controllers"]:
        evidence = controller["evidence"]
        lines.append(
            f"| {controller['label']}"
            f" | {evidence['checkpointEnvironmentVersionStatus']}"
            f" | `{evidence['checkpointSha256'] or 'n/a'}`"
            f" | `{evidence['graphArtifactSha256'] or 'n/a'}`"
            f" | `{evidence['controlGraphArtifactSha256'] or 'n/a'}` |"
        )
    lines.extend(
        [
            "",
            (
                "The rewired control matches directed degree and per-target absolute "
                "incoming normalization; it does not match every weighted-network "
                "statistic and does not isolate topology."
            ),
            "",
            "These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(config_path: str | Path, output: str | Path) -> dict[str, str]:
    """Evaluate trained policies and controls on one held-out seed suite."""
    declared_config_path = Path(config_path)
    source_path = declared_config_path.resolve()
    config = load_eval_config(source_path)
    conventional, conventional_training, conventional_environment_version = _load_checkpoint(
        config.conventional_checkpoint,
        config.conventional_checkpoint_sha256,
        "conventional",
        config.training_seeds,
        config.environment_version,
    )
    connectome, connectome_training, connectome_environment_version = _load_checkpoint(
        config.connectome_checkpoint,
        config.connectome_checkpoint_sha256,
        "connectome",
        config.training_seeds,
        config.environment_version,
    )
    concrete_training_world_seeds = set(conventional_training["world_seeds"]) | set(
        connectome_training["world_seeds"]
    )
    overlap = concrete_training_world_seeds & set(config.evaluation_seeds)
    if overlap:
        raise ValueError(
            "Evaluation concrete training world seeds overlap held-out seeds: "
            + ", ".join(sorted(overlap))
        )
    if not isinstance(
        connectome, (FixedGraphPolicy, PopulationFixedGraphPolicy, NestedPopulationFixedGraphPolicy, GatedNestedPopulationFixedGraphPolicy)
    ):
        raise ValueError("Connectome evaluation requires a fixed graph policy.")

    def policy_like(graph: ReducedGraphArtifact):
        if isinstance(connectome, FeedbackNestedPopulationFixedGraphPolicy):
            return FeedbackNestedPopulationFixedGraphPolicy(
                graph,
                connectome.core_graph,
                observation_size=connectome.sensory.in_features,
                actions=connectome.actor.out_features,
            )
        if isinstance(connectome, GatedNestedPopulationFixedGraphPolicy):
            return GatedNestedPopulationFixedGraphPolicy(
                graph,
                connectome.core_graph,
                observation_size=connectome.sensory.in_features,
                actions=connectome.actor.out_features,
            )
        if isinstance(connectome, NestedPopulationFixedGraphPolicy):
            return NestedPopulationFixedGraphPolicy(
                graph,
                connectome.core_graph,
                observation_size=connectome.sensory.in_features,
                actions=connectome.actor.out_features,
            )
        return type(connectome)(
            graph,
            observation_size=connectome.sensory.in_features,
            actions=connectome.actor.out_features,
        )

    rewired_graph = degree_preserving_rewire(
        connectome.graph, seed=config.rewiring_seed, swaps=config.rewiring_swaps
    )
    rewired = policy_like(rewired_graph)
    rewired.load_state_dict(connectome.state_dict(), strict=True)
    rewired.eval()

    silenced = policy_like(connectome.graph)
    silenced.load_state_dict(connectome.state_dict(), strict=True)
    silenced.eval()
    silenced_indices = _silenced_indices(connectome.graph, config.silencing_population)
    silenced_body_ids = (
        connectome.graph.sensory_body_ids
        if config.silencing_population == "sensory"
        else connectome.graph.readout_body_ids
    )

    untrained = policy_like(connectome.graph)
    untrained.load_state_dict(connectome.state_dict(), strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.untrained_readout_seed)
        untrained.actor.reset_parameters()
    untrained.eval()

    controllers = [
        _controller_result(
            controller_id="conventional",
            label="Conventional PPO",
            control="none",
            runner=_PolicyRunner(conventional),
            seeds=config.evaluation_seeds,
            maximum_steps=config.max_steps_per_episode,
            evidence=_evidence(
                conventional,
                config.conventional_checkpoint,
                conventional_training,
                checkpoint_environment_version=conventional_environment_version,
            ),
        ),
        _controller_result(
            controller_id="reduced-connectome",
            label="Reduced-connectome PPO",
            control="none",
            runner=_PolicyRunner(connectome),
            seeds=config.evaluation_seeds,
            maximum_steps=config.max_steps_per_episode,
            evidence=_evidence(
                connectome,
                config.connectome_checkpoint,
                connectome_training,
                checkpoint_environment_version=connectome_environment_version,
            ),
        ),
        _controller_result(
            controller_id="degree-preserving-rewired",
            label="Degree/normalization-matched rewired control",
            control=(
                f"{config.rewiring_swaps} directed double-edge swaps; "
                f"seed {config.rewiring_seed}; preserve directed degree and "
                "per-target absolute incoming normalization"
            ),
            runner=_PolicyRunner(rewired),
            seeds=config.evaluation_seeds,
            maximum_steps=config.max_steps_per_episode,
            evidence=_evidence(
                rewired,
                config.connectome_checkpoint,
                connectome_training,
                checkpoint_environment_version=connectome_environment_version,
                control_graph_hash=rewired_graph.artifact_sha256,
            ),
        ),
        _controller_result(
            controller_id=f"{config.silencing_population}-silenced",
            label=f"{config.silencing_population.capitalize()}-population silencing control",
            control=f"Zero {len(silenced_indices)} selected {config.silencing_population} cells",
            runner=_PolicyRunner(silenced, silenced_indices=silenced_indices),
            seeds=config.evaluation_seeds,
            maximum_steps=config.max_steps_per_episode,
            evidence=_evidence(
                silenced,
                config.connectome_checkpoint,
                connectome_training,
                checkpoint_environment_version=connectome_environment_version,
                silenced_body_ids=silenced_body_ids,
            ),
        ),
        _controller_result(
            controller_id="untrained-readout",
            label="Untrained-readout control",
            control=f"Seeded fresh actor readout; seed {config.untrained_readout_seed}",
            runner=_PolicyRunner(untrained),
            seeds=config.evaluation_seeds,
            maximum_steps=config.max_steps_per_episode,
            evidence=_evidence(
                untrained,
                config.connectome_checkpoint,
                connectome_training,
                checkpoint_environment_version=connectome_environment_version,
            ),
        ),
    ]

    human_recorded: dict[str, Any]
    if config.human_trace_files:
        human_recorded, human_controller = _evaluate_human_traces(
            config.human_trace_files,
            config.evaluation_seeds,
            config.max_steps_per_episode,
        )
        controllers.insert(0, human_controller)
    else:
        human_recorded = {"status": "not-present", "files": [], "episodeCount": 0}

    payload = {
        "version": 1,
        "environmentVersion": config.environment_version,
        "config": {
            "path": str(declared_config_path),
            "sha256": _sha256(source_path),
            "trainingSeeds": list(config.training_seeds),
            "evaluationSeeds": list(config.evaluation_seeds),
            "maxStepsPerEpisode": config.max_steps_per_episode,
            "concreteTrainingWorldSeeds": sorted(concrete_training_world_seeds),
        },
        "humanRecordedTraces": human_recorded,
        "controllers": controllers,
    }
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.json"
    csv_path = output_path / "metrics.csv"
    markdown_path = output_path / "summary.md"
    _write_json(metrics_path, payload)
    _write_csv(csv_path, payload)
    _write_markdown(markdown_path, payload)
    return {
        "metricsJson": str(metrics_path),
        "metricsCsv": str(csv_path),
        "markdown": str(markdown_path),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Fly Crossy controllers and deterministic controls."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _main() -> None:
    arguments = _parse_arguments()
    print(json.dumps(evaluate(arguments.config, arguments.output), indent=2))


if __name__ == "__main__":
    _main()
