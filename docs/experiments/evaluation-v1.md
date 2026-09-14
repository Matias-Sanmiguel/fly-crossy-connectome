# Controller evaluation v1

## Claim boundary

This is a bounded first-release evaluation of smoke checkpoints, not a convergence study. It reports behavior observed on the declared held-out seeds. It does not show that the reduced connectome learned a stable policy, that measured topology is beneficial, or that simulated activity is biologically faithful.

No human-recorded traces were configured for this run. No human baseline, human actions, or human neural activity were fabricated.

## Reproduction and budget

The commands were run from `python/` on CPU. GNU `timeout` imposed a 180-second wall limit on each new bounded run.

```sh
python -m pytest tests -q
timeout 180s python -m fly_crossy.train \
  --controller connectome \
  --seed smoke \
  --steps 2048 \
  --envs 4 \
  --learning-rate 0.0003 \
  --output runs/smoke-connectome \
  --device cpu
timeout 180s python -m fly_crossy.evaluate \
  --config ../configs/eval-v1.json \
  --output runs/eval-v1
```

The connectome smoke train completed in 6.75 seconds of observed wall time, and evaluation completed in 3.07 seconds. The conventional checkpoint was the existing Task 6 smoke artifact; its original wall time was not recorded. Both checkpoint metadata files report 2,048 training environment steps, four parallel environments, seed `smoke`, learning rate `0.0003`, and CPU as the resolved device. These are actual finite budgets, not convergence claims.

Evaluation used environment version 1, deterministic argmax actions, 12 seeds (`eval-v1-001` through `eval-v1-012`) disjoint from training seed `smoke`, and a maximum of 128 decisions per episode. Every learned and control policy used the identical evaluation suite. The config SHA-256 is `1f58d0760547cde4e7fc43eebd0ffb28913b296ca45bfe708bb512789efb208f`.

The generated machine-readable evidence is `runs/eval-v1/metrics.json`, `runs/eval-v1/metrics.csv`, and `runs/eval-v1/summary.md`. The `runs/` directory is intentionally local and ignored; rerun the command above to regenerate it.

## Checkpoints and controls

| Method | Parameters | Training environment steps | Checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| Conventional PPO | 28,294 | 2,048 | `37d9aded682a11c88404f14f3f8a0d09ef941efb18512a65ff4e89734e691c69` |
| Reduced-connectome PPO and controls | 30,088 | 2,048 | `343e438b928bc318f72d8426cb970291f8ff78f7d7237d45d57f03f656d45563` |

The reduced-connectome graph artifact SHA-256 is `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862`. The rewired control performed 1,296 deterministic directed double-edge swaps with seed `rewired-control-v1`, preserving every node's in-degree and out-degree; its graph artifact SHA-256 is `a89372262e78e6a41481eaca576d7337d3c9322cc5d68e4caa82968707c49f83`. The silencing control zeroed all 32 declared sensory-population activities after every state update, before actor readout and before the masked state was carried into the next recurrent step. The untrained-readout control retained trained sensory and recurrent parameters but reset the actor head with seed 9,173.

## Held-out results

Values below come from the emitted `metrics.json`; displayed decimals are rounded to three places.

| Method | Score mean / median / max | Survival steps mean / median / max | Wait frequency | Terminal reasons |
| --- | ---: | ---: | ---: | --- |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 6.000 / 6.000 / 6.000 | 0.000 | bounds 12 |
| Reduced-connectome PPO | 3.667 / 3.000 / 6.000 | 10.583 / 7.000 / 28.000 | 0.638 | train 4, water 8 |
| Degree-preserving rewired | 5.667 / 4.000 / 15.000 | 7.500 / 4.000 / 20.000 | 0.233 | train 3, vehicle 1, water 8 |
| Sensory population silenced | 5.833 / 4.000 / 15.000 | 5.833 / 4.000 / 15.000 | 0.000 | train 3, vehicle 1, water 8 |
| Untrained readout | 1.000 / 1.000 / 2.000 | 17.417 / 14.500 / 68.000 | 0.560 | train 3, vehicle 4, water 5 |

| Method | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 72 / 1.000 | 0 / 0.000 | 0 / 0.000 |
| Reduced-connectome PPO | 45 / 0.354 | 0 / 0.000 | 0 / 0.000 | 1 / 0.008 | 81 / 0.638 |
| Degree-preserving rewired | 68 / 0.756 | 0 / 0.000 | 0 / 0.000 | 1 / 0.011 | 21 / 0.233 |
| Sensory population silenced | 70 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Untrained readout | 31 / 0.148 | 61 / 0.292 | 0 / 0.000 | 0 / 0.000 | 117 / 0.560 |

Each action cell is `count / frequency`. The rewired and silenced controls have higher mean and maximum score than the trained reduced-connectome smoke checkpoint on this suite. That outcome is evidence against claiming a beneficial connectome-topology effect from this short run. The conventional checkpoint collapsed to moving left until a bounds terminal. More training, multiple training seeds, and a preregistered larger evaluation would be required before making learning-performance claims.

## Inference timing

Wall-clock inference timing includes only model decision calls and is machine-dependent. It is diagnostic rather than an environment-quality metric.

| Method | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 72 | 0.050212 | 0.697389 | 1,433.919 |
| Reduced-connectome PPO | 127 | 0.109316 | 0.860757 | 1,161.768 |
| Degree-preserving rewired | 90 | 0.080832 | 0.898138 | 1,113.414 |
| Sensory population silenced | 70 | 0.072473 | 1.035326 | 965.880 |
| Untrained readout | 209 | 0.155076 | 0.741992 | 1,347.724 |

## Source-artifact terminology

The bundled connectome manifest says `Engineered 8-channel input injection` because it describes the reused FlyDino source artifact and its upstream assumptions. This Fly Crossy controller does not use that eight-channel injection. It encodes each game observation as 370 values and learns a 370-to-80 sensory projection into the complete reduced graph. The manifest remains a provenance record for the reused bytes; the controller equation and implemented interface are documented separately in [Reduced MaleCNS controller v1](reduced-connectome-v1.md).

## Remaining extension

A live adapter over the full MaleCNS graph remains future work. It needs separately versioned graph selection/loading, compute and latency budgets, a validated stimulus interface, activity provenance, transport backpressure, disconnect handling, and new matched controls. It must not silently replace this named 80-cell experiment or imply whole-brain biological simulation.
