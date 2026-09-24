# Evaluation v1 emitted summary

Environment v10; 1 held-out seeds; maximum 3 steps per episode.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Human-recorded traces | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.333 | step-limit 1 | 0 | 0 |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 2122 |
| Reduced-connectome PPO | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 41848 |
| Degree/normalization-matched rewired control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 41848 |
| Sensory-population silencing control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 41848 |
| Untrained-readout control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 41848 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human-recorded traces | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 |
| Conventional PPO | 0 / 0.000 | 3 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Reduced-connectome PPO | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 2 / 0.667 | 0 / 0.000 |
| Degree/normalization-matched rewired control | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 2 / 0.667 | 0 / 0.000 |
| Sensory-population silencing control | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 3 / 1.000 | 0 / 0.000 |
| Untrained-readout control | 0 / 0.000 | 0 / 0.000 | 1 / 0.333 | 2 / 0.667 | 0 / 0.000 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Human-recorded traces | 0 | 0.000000 | 0.000000 | 0.000 |
| Conventional PPO | 3 | 0.000158 | 0.052700 | 18975.332 |
| Reduced-connectome PPO | 3 | 0.000333 | 0.110867 | 9019.844 |
| Degree/normalization-matched rewired control | 3 | 0.000253 | 0.084367 | 11853.022 |
| Sensory-population silencing control | 3 | 0.000415 | 0.138433 | 7223.694 |
| Untrained-readout control | 3 | 0.000321 | 0.107067 | 9339.975 |

## Evidence hashes

Config SHA-256: `3e449e1f717e621b460a21488d2e99129ab5b871b5195cff1d409374c4049eec`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Human-recorded traces | not-applicable | `n/a` | `n/a` | `n/a` |
| Conventional PPO | recorded-match | `5b2d1ed6deb644565948a02e264fb7192ec27b0aadf744999ec114fa8070640c` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
