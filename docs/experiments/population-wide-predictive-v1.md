# Population Wide Predictive v1

This controller experiment keeps Environment v9, ObservationV3, Reward v4 and
the 1000-cell recurrent topology unchanged.

Changes:

- sensory injection population: 32 -> 128 real cells from the existing 1k graph;
- all 32 historical sensory cells are retained;
- declared readout cells are excluded from the added sensory population;
- the 96 added input sites are selected from graph topology only, using
  bidirectional weighted degree, total weighted degree, outgoing weighted degree
  and body ID as deterministic tie-breakers;
- the actor/critic still read the same 16 historical readout cells;
- a training-only linear auxiliary head predicts the next ObservationV3 local
  scene for the three rows ahead (132 values) from the 16 readout activities
  plus the chosen action;
- auxiliary coefficient: 0.05;
- the auxiliary head is not saved in the inference model and is not exported.

The auxiliary target is the next observation actually produced by the
environment. No future value is supplied at inference and no collision oracle is
added. Terminal-to-reset transitions are masked from the auxiliary loss.
