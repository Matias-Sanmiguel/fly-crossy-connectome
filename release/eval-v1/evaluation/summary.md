# Evaluation v1 emitted summary

Environment v3; 12 held-out seeds; maximum 128 steps per episode.

No human-recorded traces were configured. No human baseline was fabricated.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 7.333 / 6.000 / 12.000 | 0.000 | bounds 12 | 2048 | 28294 |
| Reduced-connectome PPO | 3.500 / 3.000 / 6.000 | 28.000 / 27.000 / 48.000 | 0.851 | train 4, vehicle 7, water 1 | 2048 | 30088 |
| Degree/normalization-matched rewired control | 5.583 / 5.000 / 11.000 | 7.583 / 5.000 / 20.000 | 0.187 | train 3, vehicle 8, water 1 | 2048 | 30088 |
| Sensory-population silencing control | 6.250 / 5.000 / 14.000 | 6.250 / 5.000 / 14.000 | 0.000 | train 4, vehicle 7, water 1 | 2048 | 30088 |
| Untrained-readout control | 0.500 / 0.000 / 2.000 | 23.417 / 12.500 / 73.000 | 0.797 | train 9, vehicle 3 | 2048 | 30088 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 74 / 0.841 | 14 / 0.159 | 0 / 0.000 |
| Reduced-connectome PPO | 43 / 0.128 | 0 / 0.000 | 0 / 0.000 | 7 / 0.021 | 286 / 0.851 |
| Degree/normalization-matched rewired control | 67 / 0.736 | 0 / 0.000 | 0 / 0.000 | 7 / 0.077 | 17 / 0.187 |
| Sensory-population silencing control | 75 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Untrained-readout control | 16 / 0.057 | 41 / 0.146 | 0 / 0.000 | 0 / 0.000 | 224 / 0.797 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 88 | 0.049291 | 0.560129 | 1785.302 |
| Reduced-connectome PPO | 336 | 0.278983 | 0.830307 | 1204.374 |
| Degree/normalization-matched rewired control | 91 | 0.091664 | 1.007295 | 992.758 |
| Sensory-population silencing control | 75 | 0.069765 | 0.930200 | 1075.038 |
| Untrained-readout control | 281 | 0.212503 | 0.756240 | 1322.332 |

## Evidence hashes

Config SHA-256: `1abb2fc0f49ad4bff4f774ddf55a6af667e80e8c810eac4ef2bbb51cdc4e57fd`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Conventional PPO | recorded-match | `03640098bf3ff5e8e2164c1cd43dff0791dea6d2f51e711a8d2d7217648c2620` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f` | `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218` | `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218` |
| Sensory-population silencing control | recorded-match | `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `abd1dc3f472de7687599d60cf73ecfbc31207ede197deeae353b8f4d4cb76e9f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
