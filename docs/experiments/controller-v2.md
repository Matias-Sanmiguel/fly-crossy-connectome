# Connectome Controller V2

Controller V2 is a redesign, not a tuning pass.

- Environment v10 / ObservationV4 (517 inputs)
- 128 topology-selected sensory cells
- unchanged 1000-cell MaleCNS recurrent graph
- 128 topology-selected decision cells
- PPO actor/critic
- learned per-action short-horizon risk head
- learned per-action route-opening head
- teacher supervision for all five actions during training
- learned risk/route heads remain active in inference
- browser Safety Reflex does not plan traffic

Inference combines base actor logits with learned risk and route probabilities.
Reward v5 removes the old forward-at-all-costs imbalance by reducing progress reward,
removing generic stagnation punishment, and making unproductive WAIT more expensive
than lateral route preparation.
