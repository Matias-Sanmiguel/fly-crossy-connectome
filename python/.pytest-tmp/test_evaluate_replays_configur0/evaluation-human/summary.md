# Evaluation v1 emitted summary

Environment v3; 1 held-out seeds; maximum 3 steps per episode.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Human-recorded traces | 1.000 / 1.000 / 1.000 | 3.000 / 3.000 / 3.000 | 0.333 | step-limit 1 | 0 | 0 |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 1.000 / 1.000 / 1.000 | 0.000 | train 1 | 8 | 1534 |
| Reduced-connectome PPO | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.667 | step-limit 1 | 8 | 30088 |
| Degree/normalization-matched rewired control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.667 | step-limit 1 | 8 | 30088 |
| Sensory-population silencing control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 1.000 | step-limit 1 | 8 | 30088 |
| Untrained-readout control | 0.000 / 0.000 / 0.000 | 3.000 / 3.000 / 3.000 | 0.667 | step-limit 1 | 8 | 30088 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human-recorded traces | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 1 / 0.333 |
| Conventional PPO | 0 / 0.000 | 1 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Reduced-connectome PPO | 0 / 0.000 | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 2 / 0.667 |
| Degree/normalization-matched rewired control | 0 / 0.000 | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 2 / 0.667 |
| Sensory-population silencing control | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 3 / 1.000 |
| Untrained-readout control | 0 / 0.000 | 1 / 0.333 | 0 / 0.000 | 0 / 0.000 | 2 / 0.667 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Human-recorded traces | 0 | 0.000000 | 0.000000 | 0.000 |
| Conventional PPO | 1 | 0.000093 | 0.093400 | 10706.638 |
| Reduced-connectome PPO | 3 | 0.000271 | 0.090267 | 11078.286 |
| Degree/normalization-matched rewired control | 3 | 0.000224 | 0.074667 | 13392.857 |
| Sensory-population silencing control | 3 | 0.000314 | 0.104533 | 9566.326 |
| Untrained-readout control | 3 | 0.000243 | 0.080933 | 12355.848 |

## Evidence hashes

Config SHA-256: `784953803a641b5ccb12b0054dbb029a46100fd590ebc600f4d20bf4541d0bf8`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Human-recorded traces | not-applicable | `n/a` | `n/a` | `n/a` |
| Conventional PPO | recorded-match | `ce8327ee0f71ee0336d4f1a72c4cab2190a5b52b9c231bb52af7a9c08b1556e7` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
