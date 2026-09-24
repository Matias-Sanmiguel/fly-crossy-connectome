# Evaluation v1 emitted summary

Environment v9; 1 held-out seeds; maximum 3 steps per episode.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Human-recorded traces | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.333 | step-limit 1 | 0 | 0 |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 2022 |
| Reduced-connectome PPO | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 39848 |
| Degree/normalization-matched rewired control | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 39848 |
| Sensory-population silencing control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 39848 |
| Untrained-readout control | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.000 | step-limit 1 | 8 | 39848 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human-recorded traces | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 2 / 0.667 | 1 / 0.333 | 0 / 0.000 |
| Reduced-connectome PPO | 1 / 0.333 | 0 / 0.000 | 0 / 0.000 | 2 / 0.667 | 0 / 0.000 |
| Degree/normalization-matched rewired control | 1 / 0.333 | 0 / 0.000 | 0 / 0.000 | 2 / 0.667 | 0 / 0.000 |
| Sensory-population silencing control | 0 / 0.000 | 0 / 0.000 | 1 / 0.333 | 2 / 0.667 | 0 / 0.000 |
| Untrained-readout control | 1 / 0.333 | 0 / 0.000 | 2 / 0.667 | 0 / 0.000 | 0 / 0.000 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Human-recorded traces | 0 | 0.000000 | 0.000000 | 0.000 |
| Conventional PPO | 3 | 0.000266 | 0.088800 | 11261.261 |
| Reduced-connectome PPO | 3 | 0.000458 | 0.152633 | 6551.649 |
| Degree/normalization-matched rewired control | 3 | 0.000449 | 0.149533 | 6687.472 |
| Sensory-population silencing control | 3 | 0.000450 | 0.149900 | 6671.114 |
| Untrained-readout control | 3 | 0.000408 | 0.136167 | 7343.941 |

## Evidence hashes

Config SHA-256: `b35108f3135be74a56d7a41c8f33cef59028c1787646849e5136ad9ec97c9c77`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Human-recorded traces | not-applicable | `n/a` | `n/a` | `n/a` |
| Conventional PPO | recorded-match | `19eeffe14da969b4481626c893d211b4504de089188712759043ade70e36fb27` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
