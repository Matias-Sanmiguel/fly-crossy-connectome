# Environment v11 + training-only expert planner

## Non-negotiable inference boundary

The final expo controller remains the MaleCNS neural network.

The expert planner in `python/fly_crossy/expert_planner.py` is only:

- a physics validator;
- an offline teacher for future imitation-learning data;
- a benchmark before spending GPU time.

It is not imported by the browser controller, is not serialized into
`policy.json`, and must never choose actions during the expo.

## Environment v11 collision timing

V10 checked whether traffic swept the fly's starting cell for the full 0.2 s
decision interval before applying the requested action. A fly that visually
escaped a road could therefore still die in the old cell.

V11 uses action-first semantics:

1. apply the discrete action;
2. blocked moves remain in place;
3. advance traffic for the 0.2 s interval;
4. test swept collision on the destination cell.

WAIT and blocked moves remain exposed. Entering a road is still unsafe if a
vehicle sweeps the destination during the interval.

## Expert planner

The planner uses only authoritative `step_game` transitions. It performs
receding-horizon search and executes one action at a time. Death inside the
search horizon is worse than every surviving branch; among surviving branches
it prefers score progress, route flexibility and centrality.

The planner deliberately ignores the PPO reward. Its job is to establish that
the environment itself admits sensible traffic and river decisions before it is
used to label neural-network training data.


## Renderer clock synchronization

The renderer previously received a post-step state at time `t1` but animated
hazards from `t1` toward `t1 + 0.2` while the fly visually replayed the hop
from the previous position to the new one. The image was therefore one
decision interval ahead of the collision interval used by the simulation.

V11 now interpolates hazards over the same authoritative transition `t0 -> t1`
while the fly visually moves between those same two states.
