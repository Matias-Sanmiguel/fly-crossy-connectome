# Environment v6 final 80-neuron release

Date: 2026-09-16

## Identity

- Environment: v6
- Frozen environment commit: `486a8265cf793daf9e1ea024b1c147b42a7370b3`
- Population: 80 neurons
- Controller: reduced MaleCNS-derived fixed graph
- Training algorithm: PPO
- Selected run: `final-80n-v6-train-1m-01`
- Training steps: 1,000,000
- Parallel environments: 8
- Learning rate: 0.0003
- Reward: v2
- Graph SHA-256: `2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719`
- Checkpoint SHA-256: `0170f778031161590761cac4ec87a0815d555051cac826c5019dc5bda469d8d9`

## Model selection

Three independent 1,000,000-step runs were evaluated on the same 100 validation
seeds (`heldout-v6-000` through `heldout-v6-099`) with a 200-step limit.

The selected run was chosen before the final-test seeds were inspected, using
highest validation mean score as the predeclared selection criterion.

| Run | Score mean | Median | Max | Survival mean | Step-limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| final-80n-v6-train-1m-01 | 23.33 | 21.5 | 84.0 | 126.21 | 46/100 |
| final-80n-v6-train-1m-02 | 16.54 | 11.0 | 73.0 | 76.04 | 17/100 |
| final-80n-v6-train-1m-03 | 19.88 | 16.0 | 65.0 | 121.76 | 38/100 |

## Final untouched test

The selected checkpoint was evaluated once on 200 new seeds
(`final-test-v6-000` through `final-test-v6-199`) with a 500-step limit.

- score mean: 19.13
- score median: 12.5
- score max: 82.0
- survival mean: 187.63
- survival median: 87.5
- survival max: 500.0
- step-limit: 55/200 (27.5%)
- vehicle: 88/200 (44.0%)
- train: 52/200 (26.0%)
- water: 5/200 (2.5%)

Action distribution:

- forward: 0.158690
- backward: 0.021585
- left: 0.477669
- right: 0.316847
- wait: 0.025209

## Interpretation boundary

This is the final held-out performance record for the selected 80-neuron v6
controller. The final-test seeds must not be reused for model selection,
hyperparameter tuning, reward changes, or environment changes.

This controller is a reduced MaleCNS-derived fixed-graph model with engineered
sensory/readout interfaces. It is not a full biological simulation and this
release does not establish a beneficial connectome-topology effect.
