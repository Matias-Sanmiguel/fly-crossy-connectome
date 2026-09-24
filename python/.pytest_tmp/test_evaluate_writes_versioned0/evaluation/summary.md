# Evaluation v1 emitted summary

Environment v9; 2 held-out seeds; maximum 4 steps per episode.

No human-recorded traces were configured. No human baseline was fabricated.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 2022 |
| Reduced-connectome PPO | 1.000 / 1.000 / 2.000 | 4.000 / 4.000 / 4.000 | 0.125 | step-limit 2 | 8 | 39848 |
| Degree/normalization-matched rewired control | 1.000 / 1.000 / 2.000 | 4.000 / 4.000 / 4.000 | 0.125 | step-limit 2 | 8 | 39848 |
| Sensory-population silencing control | 0.500 / 0.500 / 1.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 39848 |
| Untrained-readout control | 2.000 / 2.000 / 3.000 | 4.000 / 4.000 / 4.000 | 0.125 | step-limit 2 | 8 | 39848 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 0 / 0.000 | 0 / 0.000 | 5 / 0.625 | 3 / 0.375 | 0 / 0.000 |
| Reduced-connectome PPO | 2 / 0.250 | 0 / 0.000 | 4 / 0.500 | 1 / 0.125 | 1 / 0.125 |
| Degree/normalization-matched rewired control | 2 / 0.250 | 0 / 0.000 | 4 / 0.500 | 1 / 0.125 | 1 / 0.125 |
| Sensory-population silencing control | 1 / 0.125 | 0 / 0.000 | 3 / 0.375 | 4 / 0.500 | 0 / 0.000 |
| Untrained-readout control | 4 / 0.500 | 0 / 0.000 | 1 / 0.125 | 2 / 0.250 | 1 / 0.125 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 8 | 0.001408 | 0.176062 | 5679.801 |
| Reduced-connectome PPO | 8 | 0.000984 | 0.122938 | 8134.215 |
| Degree/normalization-matched rewired control | 8 | 0.000746 | 0.093200 | 10729.614 |
| Sensory-population silencing control | 8 | 0.001042 | 0.130212 | 7679.754 |
| Untrained-readout control | 8 | 0.000764 | 0.095475 | 10473.946 |

## Evidence hashes

Config SHA-256: `2ee0bd0bba1d59cfe6a0676e29b6ef1994c072a4b44c9d12f2dfeb1f9ce3b7b0`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Conventional PPO | recorded-match | `19eeffe14da969b4481626c893d211b4504de089188712759043ade70e36fb27` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `8b5731340cfc9268d743c68f668f54054f0e26e8a113842bdb6ed6cf91e70f2f` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
