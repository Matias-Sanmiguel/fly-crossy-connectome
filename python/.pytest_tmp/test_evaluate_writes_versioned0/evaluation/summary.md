# Evaluation v1 emitted summary

Environment v10; 2 held-out seeds; maximum 4 steps per episode.

No human-recorded traces were configured. No human baseline was fabricated.

| Controller / control | Score mean / median / max | Survival mean / median / max | Wait frequency | Terminal reasons | Training steps | Parameters |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Conventional PPO | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 2122 |
| Reduced-connectome PPO | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 41848 |
| Degree/normalization-matched rewired control | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 41848 |
| Sensory-population silencing control | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 41848 |
| Untrained-readout control | 0.000 / 0.000 / 0.000 | 4.000 / 4.000 / 4.000 | 0.000 | step-limit 2 | 8 | 41848 |

## Action distributions

Each action cell is `count / frequency`.

| Controller / control | Forward | Backward | Left | Right | Wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conventional PPO | 1 / 0.125 | 5 / 0.625 | 0 / 0.000 | 2 / 0.250 | 0 / 0.000 |
| Reduced-connectome PPO | 0 / 0.000 | 0 / 0.000 | 1 / 0.125 | 7 / 0.875 | 0 / 0.000 |
| Degree/normalization-matched rewired control | 0 / 0.000 | 0 / 0.000 | 1 / 0.125 | 7 / 0.875 | 0 / 0.000 |
| Sensory-population silencing control | 0 / 0.000 | 0 / 0.000 | 0 / 0.000 | 8 / 1.000 | 0 / 0.000 |
| Untrained-readout control | 0 / 0.000 | 0 / 0.000 | 4 / 0.500 | 4 / 0.500 | 0 / 0.000 |

## Inference performance

Wall-clock values include model decision calls only and are machine-dependent.

| Controller / control | Calls | Wall seconds | Mean milliseconds | Calls per second |
| --- | ---: | ---: | ---: | ---: |
| Conventional PPO | 8 | 0.000299 | 0.037388 | 26746.907 |
| Reduced-connectome PPO | 8 | 0.000749 | 0.093650 | 10678.057 |
| Degree/normalization-matched rewired control | 8 | 0.000626 | 0.078250 | 12779.553 |
| Sensory-population silencing control | 8 | 0.000803 | 0.100313 | 9968.847 |
| Untrained-readout control | 8 | 0.000786 | 0.098212 | 10182.003 |

## Evidence hashes

Config SHA-256: `92caf8cea08aed76984e108045dd618acfa7105657db70b1f1764c4f91e5d6ba`.

| Controller / control | Checkpoint environment provenance | Checkpoint SHA-256 | Graph artifact SHA-256 | Control graph artifact SHA-256 |
| --- | --- | --- | --- | --- |
| Conventional PPO | recorded-match | `5b2d1ed6deb644565948a02e264fb7192ec27b0aadf744999ec114fa8070640c` | `n/a` | `n/a` |
| Reduced-connectome PPO | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Degree/normalization-matched rewired control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` | `751c8c0713a45657adbb68dc48869a8fa98edbc2738f29db20a6ce02da1aa9b4` |
| Sensory-population silencing control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |
| Untrained-readout control | recorded-match | `91b4219787807c17bf46dd0169c8598d537c0247e6a89243153288938e3c85e1` | `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862` | `n/a` |

The rewired control matches directed degree and per-target absolute incoming normalization; it does not match every weighted-network statistic and does not isolate topology.

These bounded runs report observed smoke-checkpoint behavior only; they are not convergence or biological-superiority claims.
