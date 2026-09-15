# Evaluation v1 emitted summary

Environment v3; 2 held-out seeds; maximum 4 steps per episode.

No human-recorded traces were configured. No human baseline was fabricated.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 1534 |
| Reduced-connectome PPO | 1.500 / 1.500 / 3.000 | 4.000 / 4.000 / 4.000 | 0.625 | step-limit 1, train 1 | 8 | 30088 |
| Degree/normalization-matched rewired control | 1.500 / 1.500 / 3.000 | 4.000 / 4.000 / 4.000 | 0.625 | step-limit 1, train 1 | 8 | 30088 |
| Sensory-population silencing control | 1.500 / 1.500 / 3.000 | 4.000 / 4.000 / 4.000 | 0.500 | step-limit 1, train 1 | 8 | 30088 |
| Untrained-readout control | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 1.000 | step-limit 2 | 8 | 30088 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 8 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Reduced-connectome PPO | 3 / 0.375 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 5 / 0.625 |
| Degree/normalization-matched rewired control | 3 / 0.375 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 5 / 0.625 |
| Sensory-population silencing control | 3 / 0.375 | 0 / 0.000 | 1 / 0.125 | 0 / 0.000 | 4 / 0.500 |
| Untrained-readout control | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 8 / 1.000 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 8 | 0.000975 | 0.121900 | 8203.446 |
| Reduced-connectome PPO | 8 | 0.000700 | 0.087525 | 11425.307 |
| Degree/normalization-matched rewired control | 8 | 0.000538 | 0.067238 | 14872.653 |
| Sensory-population silencing control | 8 | 0.000729 | 0.091175 | 10967.919 |
| Untrained-readout control | 8 | 0.000540 | 0.067500 | 14814.815 |

## Evidence hashes

Config SHA-256: `8e895a3cb25f895e9633f85123bff861719a12452333e2db6e6852dee030e39e`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Conventional PPO | recorded-match | `ce8327ee0f71ee0336d4f1a72c4cab2190a5b52b9c231bb52af7a9c08b1556e7` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `18f23991da157b1cc3715ce8e4586469fcdd282912f7f57341c5a411dd37951a` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
