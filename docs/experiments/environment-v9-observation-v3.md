# Environment v9 / ObservationV3

Environment v9 keeps the v6 world layout and Reward v4 unchanged. The only
behavioral interface change is ObservationV3.

ObservationV3 has 492 float inputs, in this order:

1. 121 normalized cell codes (`cell / 8`)
2. 242 motion values (`direction`, normalized speed)
3. 121 continuous hazard-offset values
4. 1 river-support bit
5. 5 previous-action one-hot values
6. 1 unsigned nearest-edge distance
7. 1 signed lateral position (`column / 5`)

`hazardOffset` is only present for a hazard that currently occupies the sampled
cell. It is the current hazard-center offset from that cell, normalized by the
hazard half-size and clamped to [-1, 1]. It contains current sub-cell phase, not
future state or a collision oracle.

`signedColumn` is proprioceptive lateral position: -1 is the left edge, 0 the
center, and +1 the right edge.

The purpose of v9 is to remove two information bottlenecks in ObservationV2:
left/right ambiguity from the unsigned edge-distance scalar and loss of
within-cell traffic phase. World generation, actions, transition physics,
collision rules and Reward v4 are unchanged.
