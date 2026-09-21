from __future__ import annotations

from pathlib import Path

import torch
from torch.distributions import Categorical

from fly_crossy.connectome import load_reduced_graph_variant
from fly_crossy.env import hash_seed
from fly_crossy.models import PopulationFixedGraphPolicy
from fly_crossy.schema import ACTION_ORDER, OBSERVATION_INPUT_SIZE
from fly_crossy.train import (
    RECURRENT_TRAINING_VERSION,
    TrainingConfig,
    _ppo_update_recurrent,
    train,
)


def _population(seed: str) -> PopulationFixedGraphPolicy:
    torch.manual_seed(hash_seed(seed))
    return PopulationFixedGraphPolicy(
        load_reduced_graph_variant("80"),
        OBSERVATION_INPUT_SIZE,
        len(ACTION_ORDER),
    )


def test_sequence_ppo_backpropagates_into_population_sensory_projection() -> None:
    seed = "recurrent-bptt-gradient"
    model = _population(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    time_steps = 4
    observations = torch.randn(time_steps, 1, OBSERVATION_INPUT_SIZE)
    actions = torch.tensor([[0], [1], [2], [4]], dtype=torch.long)
    dones = torch.zeros(time_steps, 1)
    initial_hidden = torch.zeros(1, model.graph.node_count)

    old_log_probabilities: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    hidden = initial_hidden.clone()
    with torch.no_grad():
        for timestep in range(time_steps):
            logits, values, hidden = model(observations[timestep], hidden)
            old_log_probabilities.append(
                Categorical(logits=logits).log_prob(actions[timestep])
            )
            old_values.append(values)

    before = model.sensory.weight.detach().clone()

    _ppo_update_recurrent(
        model,
        optimizer,
        observations,
        actions,
        torch.stack(old_log_probabilities),
        torch.tensor([[1.0], [-0.5], [0.75], [-1.25]]),
        torch.stack(old_values),
        dones,
        initial_hidden,
    )

    after = model.sensory.weight.detach()
    assert not torch.equal(after, before)
    assert float(torch.linalg.vector_norm(after - before)) > 0.0


def test_population_tiny_training_changes_sensory_weights_and_records_trainer(
    tmp_path: Path,
) -> None:
    seed = "population-bptt-tiny"
    initial = _population(seed)
    initial_sensory = initial.sensory.weight.detach().clone()

    output = tmp_path / "population-bptt"
    train(
        TrainingConfig(
            controller="connectome",
            seed=seed,
            steps=64,
            envs=1,
            learning_rate=3e-4,
            output=output,
            device="cpu",
            connectome_graph="80",
            connectome_interface="population",
        )
    )

    saved = torch.load(
        output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    trained_sensory = saved["model_state_dict"]["sensory.weight"]

    assert not torch.equal(trained_sensory, initial_sensory)
    assert float(
        torch.linalg.vector_norm(trained_sensory - initial_sensory)
    ) > 0.0
    assert (
        saved["training"]["recurrent_training"]
        == RECURRENT_TRAINING_VERSION
    )
