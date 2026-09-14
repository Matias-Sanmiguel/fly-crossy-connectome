# Controller evaluation v1

## Claim boundary

This is a bounded first-release evaluation of smoke checkpoints, not a convergence study. It reports behavior observed on the declared held-out seeds. It does not show that the reduced connectome learned a stable policy, that measured topology is beneficial, or that simulated activity is biologically faithful.

No human-recorded traces were configured for this run. No human baseline, human actions, or human neural activity were fabricated.

## Reproduction and budget

The commands were run from `python/` on CPU. GNU `timeout` imposed a 180-second wall limit on each bounded run. Python 3.14.7 is recorded in `.python-version`. The declared Linux x86_64 build is described by `python/release-environment-linux-x86_64.json`, and its complete active dependency closure is exactly version-pinned in `python/requirements-linux-x86_64-cu130.txt`.

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple pip==26.2.1
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-linux-x86_64-cu130.txt
python -m pip install -e . --no-deps --no-build-isolation
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

Both checkpoints were freshly retrained with the current trainer and environment v3. The conventional and reduced-connectome commands completed in 7.7 and 9.0 seconds respectively in the recorded release run; the final evaluation completed in 4.1 seconds. Both checkpoint metadata files report 2,048 training environment steps, four parallel environments, seed `smoke`, learning rate `0.0003`, and CPU as the resolved device. These are actual finite budgets, not convergence claims.

Evaluation used environment version 3, deterministic argmax actions, 12 seeds (`eval-v1-001` through `eval-v1-012`), and a maximum of 128 decisions per episode. Every learned and control policy used the identical evaluation suite. The evaluator checked all 93 distinct concrete training-world seeds recorded across both checkpoints—not only the root `smoke` label—and found no overlap with the held-out suite. The config SHA-256 is `1abb2fc0f49ad4bff4f774ddf55a6af667e80e8c810eac4ef2bbb51cdc4e57fd`.

Both checkpoints record environment version 3. The evaluator fails closed if a checkpoint is missing that version, differs from the config, or does not match the lowercase SHA-256 declared in the config. It performs the hash check before deserialization, then strictly validates raw mappings, integer types, model/training metadata, and finite tensors before model construction.

Environment v3 replaces the v2 group solver's integer time ticks with the exact floating-point transition used by live play. Accepted witness actions are replay-tested through the authoritative JS and Python steps over a bounded multiseed/group sample; generation still uses deterministic retries and a conservative grass fallback.

This is not a hermetic wheel-hash lock: installed metadata preserved exact distribution versions and the torch build, but not wheel filenames or hashes. The release environment manifest declares that boundary explicitly. Reproduction commands are intended for that Linux x86_64/CPython build and do not claim cross-platform checkpoint byte identity. The tracked checkpoint hashes remain the authoritative identity of the released bytes.

The tracked release is [`release/eval-v1/`](../../release/eval-v1/): it contains both training bundles and the machine-readable `evaluation/metrics.json`, `evaluation/metrics.csv`, and `evaluation/summary.md`. [`manifest.json`](../../release/eval-v1/manifest.json) pins every released file. The `python/runs/` directory remains ignored scratch space for clean-checkout reproduction.

## Checkpoints and controls

| Method | Parameters | Training environment steps | Environment provenance | Checkpoint SHA-256 |
| --- | ---: | ---: | --- | --- |
| Conventional PPO | 28,294 | 2,048 | `recorded-match` (v3) | `03640098bf3ff5e8e2164c1cd43dff0791dea6d2f51e711a8d2d7217648c2620` |
| Reduced-connectome PPO and controls | 30,088 | 2,048 | `recorded-match` (v3) | `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f` |

The reduced-connectome graph artifact SHA-256 is `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862`. The rewired control performed 1,296 deterministic directed double-edge swaps with seed `rewired-control-v1`, preserving every node's in-degree and out-degree, then renormalized signed weights to preserve each connected target's absolute incoming sum. Its graph artifact SHA-256 is `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218`.

This is a matched directed-degree and incoming-normalization control, not a complete weighted-network match. Renormalization changes individual weight magnitudes, and target swaps do not preserve every higher-order or weighted statistic, so differences cannot be attributed to topology alone. The silencing control zeroed all 32 declared sensory-population activities after every state update, before actor readout and before the masked state was carried into the next recurrent step. The untrained-readout control retained trained sensory and recurrent parameters but reset the actor head with seed 9,173.

## Held-out results

Values below come from the emitted `metrics.json`; displayed decimals are rounded to three places.

| Method | Score mean / median / max | Survival steps mean / median / max | Wait frequency | Terminal reasons |
| --- | ---: | ---: | ---: | --- |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 7.333 / 6.000 / 12.000 | 0.000 | bounds 12 |
| Reduced-connectome PPO | 3.500 / 3.000 / 6.000 | 28.000 / 27.000 / 48.000 | 0.851 | train 4, vehicle 7, water 1 |
| Degree/normalization-matched rewired | 5.583 / 5.000 / 11.000 | 7.583 / 5.000 / 20.000 | 0.187 | train 3, vehicle 8, water 1 |
| Sensory population silenced | 6.250 / 5.000 / 14.000 | 6.250 / 5.000 / 14.000 | 0.000 | train 4, vehicle 7, water 1 |
| Untrained readout | 0.500 / 0.000 / 2.000 | 23.417 / 12.500 / 73.000 | 0.797 | train 9, vehicle 3 |

| Method | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 74 / 0.841 | 14 / 0.159 | 0 / 0.000 |
| Reduced-connectome PPO | 43 / 0.128 | 0 / 0.000 | 0 / 0.000 | 7 / 0.021 | 286 / 0.851 |
| Degree/normalization-matched rewired | 67 / 0.736 | 0 / 0.000 | 0 / 0.000 | 7 / 0.077 | 17 / 0.187 |
| Sensory population silenced | 75 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Untrained readout | 16 / 0.057 | 41 / 0.146 | 0 / 0.000 | 0 / 0.000 | 224 / 0.797 |

Each action cell is `count / frequency`. The matched rewired and silenced controls have higher mean and maximum score than the trained reduced-connectome smoke checkpoint on this suite. That outcome is evidence against claiming a beneficial connectome-topology effect from this short run, but it does not isolate a causal topology effect. The conventional checkpoint alternated left/right decisions but still reached a bounds terminal in every episode. More training, multiple training seeds, and a preregistered larger evaluation would be required before making learning-performance claims.

## Inference timing

Wall-clock inference timing includes only model decision calls and is machine-dependent. It is diagnostic rather than an environment-quality metric.

| Method | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 88 | 0.049291 | 0.560129 | 1,785.302 |
| Reduced-connectome PPO | 336 | 0.278983 | 0.830307 | 1,204.374 |
| Degree/normalization-matched rewired | 91 | 0.091664 | 1.007295 | 992.758 |
| Sensory population silenced | 75 | 0.069765 | 0.930200 | 1,075.038 |
| Untrained readout | 281 | 0.212503 | 0.756240 | 1,322.332 |

## Source-artifact terminology

The bundled connectome manifest says `Engineered 8-channel input injection` because it describes the reused FlyDino source artifact and its upstream assumptions. This Fly Crossy controller does not use that eight-channel injection. It encodes each game observation as 370 values and learns a 370-to-80 sensory projection into the complete reduced graph. The manifest remains a provenance record for the reused bytes; the controller equation and implemented interface are documented separately in [Reduced MaleCNS controller v1](reduced-connectome-v1.md).

## Remaining extension

A live adapter over the full MaleCNS graph remains future work. It needs separately versioned graph selection/loading, compute and latency budgets, a validated stimulus interface, activity provenance, transport backpressure, disconnect handling, and new matched controls. It must not silently replace this named 80-cell experiment or imply whole-brain biological simulation.
