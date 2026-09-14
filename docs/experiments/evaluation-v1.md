# Controller evaluation v1

## Claim boundary

This is a bounded first-release evaluation of smoke checkpoints, not a convergence study. It reports behavior observed on the declared held-out seeds. It does not show that the reduced connectome learned a stable policy, that measured topology is beneficial, or that simulated activity is biologically faithful.

No human-recorded traces were configured for this run. No human baseline, human actions, or human neural activity were fabricated.

## Reproduction and budget

The commands were run from `python/` on CPU. GNU `timeout` imposed a 180-second wall limit on each bounded run. Python 3.14.7 and the exact package set are pinned by `.python-version` and `python/requirements-lock.txt`.

```sh
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest tests -q
timeout 180s python -m fly_crossy.train \
  --controller conventional \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/reproduce-conventional \
  --device cpu
timeout 180s python -m fly_crossy.train \
  --controller connectome \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/reproduce-connectome \
  --device cpu
timeout 180s python -m fly_crossy.evaluate \
  --config ../configs/eval-v1.json \
  --output runs/eval-v1
```

Both checkpoints were freshly retrained with the current trainer and environment v2. The two bounded training commands completed in 26.9 seconds combined in the recorded release run; the final evaluation completed in 5.0 seconds. Both checkpoint metadata files report 2,048 training environment steps, four parallel environments, seed `smoke`, learning rate `0.0003`, and CPU as the resolved device. These are actual finite budgets, not convergence claims.

Evaluation used environment version 2, deterministic argmax actions, 12 seeds (`eval-v1-001` through `eval-v1-012`), and a maximum of 128 decisions per episode. Every learned and control policy used the identical evaluation suite. The evaluator checked all 118 concrete training-world seeds recorded across both checkpoints—not only the root `smoke` label—and found no overlap with the held-out suite. The config SHA-256 is `7eeca30f06824330b8287ce2a23088765d0eba456c08df93d7f044c14c62234a`.

Both checkpoints record environment version 2. The evaluator fails closed if a checkpoint is missing that version, differs from the config, or does not match the lowercase SHA-256 declared in the config. It performs the hash check before deserialization.

The tracked release is [`release/eval-v1/`](../../release/eval-v1/): it contains both training bundles and the machine-readable `evaluation/metrics.json`, `evaluation/metrics.csv`, and `evaluation/summary.md`. [`manifest.json`](../../release/eval-v1/manifest.json) pins every released file. The `python/runs/` directory remains ignored scratch space for clean-checkout reproduction.

## Checkpoints and controls

| Method | Parameters | Training environment steps | Environment provenance | Checkpoint SHA-256 |
| --- | ---: | ---: | --- | --- |
| Conventional PPO | 28,294 | 2,048 | `recorded-match` (v2) | `c336a53086e765e242e74ec396225f51d744679aff8fe1ae7fe8c31e78b2ed4d` |
| Reduced-connectome PPO and controls | 30,088 | 2,048 | `recorded-match` (v2) | `218d319c050983c225c943ab8de7493af672de561749a1892aa507a1c676cec5` |

The reduced-connectome graph artifact SHA-256 is `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862`. The rewired control performed 1,296 deterministic directed double-edge swaps with seed `rewired-control-v1`, preserving every node's in-degree and out-degree, then renormalized signed weights to preserve each connected target's absolute incoming sum. Its graph artifact SHA-256 is `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218`.

This is a matched directed-degree and incoming-normalization control, not a complete weighted-network match. Renormalization changes individual weight magnitudes, and target swaps do not preserve every higher-order or weighted statistic, so differences cannot be attributed to topology alone. The silencing control zeroed all 32 declared sensory-population activities after every state update, before actor readout and before the masked state was carried into the next recurrent step. The untrained-readout control retained trained sensory and recurrent parameters but reset the actor head with seed 9,173.

## Held-out results

Values below come from the emitted `metrics.json`; displayed decimals are rounded to three places.

| Method | Score mean / median / max | Survival steps mean / median / max | Wait frequency | Terminal reasons |
| --- | ---: | ---: | ---: | --- |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 16.167 / 9.000 / 50.000 | 0.000 | bounds 12 |
| Reduced-connectome PPO | 6.500 / 5.000 / 13.000 | 11.583 / 10.000 / 28.000 | 0.374 | train 5, vehicle 5, water 2 |
| Degree/normalization-matched rewired | 8.333 / 5.000 / 28.000 | 8.583 / 5.000 / 29.000 | 0.019 | train 5, vehicle 4, water 3 |
| Sensory population silenced | 8.333 / 5.000 / 28.000 | 8.333 / 5.000 / 28.000 | 0.000 | train 5, vehicle 4, water 3 |
| Untrained readout | 0.917 / 1.000 / 2.000 | 20.417 / 16.000 / 54.000 | 0.624 | train 4, vehicle 5, water 3 |

| Method | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 91 / 0.469 | 103 / 0.531 | 0 / 0.000 |
| Reduced-connectome PPO | 79 / 0.568 | 0 / 0.000 | 0 / 0.000 | 8 / 0.058 | 52 / 0.374 |
| Degree/normalization-matched rewired | 100 / 0.971 | 0 / 0.000 | 0 / 0.000 | 1 / 0.010 | 2 / 0.019 |
| Sensory population silenced | 100 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Untrained readout | 29 / 0.118 | 63 / 0.257 | 0 / 0.000 | 0 / 0.000 | 153 / 0.624 |

Each action cell is `count / frequency`. The matched rewired and silenced controls have higher mean and maximum score than the trained reduced-connectome smoke checkpoint on this suite. That outcome is evidence against claiming a beneficial connectome-topology effect from this short run, but it does not isolate a causal topology effect. The conventional checkpoint alternated left/right decisions but still reached a bounds terminal in every episode. More training, multiple training seeds, and a preregistered larger evaluation would be required before making learning-performance claims.

## Inference timing

Wall-clock inference timing includes only model decision calls and is machine-dependent. It is diagnostic rather than an environment-quality metric.

| Method | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 194 | 0.061821 | 0.318666 | 3,138.084 |
| Reduced-connectome PPO | 139 | 0.051263 | 0.368801 | 2,711.490 |
| Degree/normalization-matched rewired | 103 | 0.041846 | 0.406269 | 2,461.425 |
| Sensory population silenced | 100 | 0.042550 | 0.425502 | 2,350.163 |
| Untrained readout | 245 | 0.093648 | 0.382235 | 2,616.190 |

## Source-artifact terminology

The bundled connectome manifest says `Engineered 8-channel input injection` because it describes the reused FlyDino source artifact and its upstream assumptions. This Fly Crossy controller does not use that eight-channel injection. It encodes each game observation as 370 values and learns a 370-to-80 sensory projection into the complete reduced graph. The manifest remains a provenance record for the reused bytes; the controller equation and implemented interface are documented separately in [Reduced MaleCNS controller v1](reduced-connectome-v1.md).

## Remaining extension

A live adapter over the full MaleCNS graph remains future work. It needs separately versioned graph selection/loading, compute and latency budgets, a validated stimulus interface, activity provenance, transport backpressure, disconnect handling, and new matched controls. It must not silently replace this named 80-cell experiment or imply whole-brain biological simulation.
