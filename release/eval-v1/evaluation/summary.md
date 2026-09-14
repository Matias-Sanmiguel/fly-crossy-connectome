# Evaluation v1 emitted summary

Environment v2; 12 held-out seeds; maximum 128 steps per episode.

No human-recorded traces were configured. No human baseline was fabricated.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 16.167 / 9.000 / 50.000 | 0.000 | bounds 12 | 2048 | 28294 |
| Reduced-connectome PPO | 6.500 / 5.000 / 13.000 | 11.583 / 10.000 / 28.000 | 0.374 | train 5, vehicle 5, water 2 | 2048 | 30088 |
| Degree/normalization-matched rewired control | 8.333 / 5.000 / 28.000 | 8.583 / 5.000 / 29.000 | 0.019 | train 5, vehicle 4, water 3 | 2048 | 30088 |
| Sensory-population silencing control | 8.333 / 5.000 / 28.000 | 8.333 / 5.000 / 28.000 | 0.000 | train 5, vehicle 4, water 3 | 2048 | 30088 |
| Untrained-readout control | 0.917 / 1.000 / 2.000 | 20.417 / 16.000 / 54.000 | 0.624 | train 4, vehicle 5, water 3 | 2048 | 30088 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 91 / 0.469 | 103 / 0.531 | 0 / 0.000 |
| Reduced-connectome PPO | 79 / 0.568 | 0 / 0.000 | 0 / 0.000 | 8 / 0.058 | 52 / 0.374 |
| Degree/normalization-matched rewired control | 100 / 0.971 | 0 / 0.000 | 0 / 0.000 | 1 / 0.010 | 2 / 0.019 |
| Sensory-population silencing control | 100 / 1.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| Untrained-readout control | 29 / 0.118 | 63 / 0.257 | 0 / 0.000 | 0 / 0.000 | 153 / 0.624 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 194 | 0.061821 | 0.318666 | 3138.084 |
| Reduced-connectome PPO | 139 | 0.051263 | 0.368801 | 2711.490 |
| Degree/normalization-matched rewired control | 103 | 0.041846 | 0.406269 | 2461.425 |
| Sensory-population silencing control | 100 | 0.042550 | 0.425502 | 2350.163 |
| Untrained-readout control | 245 | 0.093648 | 0.382235 | 2616.190 |

## Evidence hashes

Config SHA-256: `7eeca30f06824330b8287ce2a23088765d0eba456c08df93d7f044c14c62234a`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Conventional PPO | recorded-match | `c336a53086e765e242e74ec396225f51d744679aff8fe1ae7fe8c31e78b2ed4d` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `218d319c050983c225c943ab8de7493af672de561749a1892aa507a1c676cec5` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `218d319c050983c225c943ab8de7493af672de561749a1892aa507a1c676cec5` | `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218` | `6cad0da5dc79c14ca9ddfe7040525a2495f3c7c7ec40c3f28fe5ad107e13c218` |
| Sensory-population silencing control | recorded-match | `218d319c050983c225c943ab8de7493af672de561749a1892aa507a1c676cec5` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `218d319c050983c225c943ab8de7493af672de561749a1892aa507a1c676cec5` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
