# Imitation Controller V1

The final expo controller remains the 1k MaleCNS neural network.

## Development-only expert

`expert_planner.py` is used only to validate physics, generate labelled expert
trajectories, and later label DAgger states produced by the neural policy.

The planner is never serialized into `policy.json`, never imported by the
browser controller, and never chooses an expo action.

## Dataset

`expert_dataset.py` stores complete expert episodes, not shuffled independent
frames. Episode offsets preserve temporal boundaries for recurrent training.

## Neural architecture

    ObservationV4 (517)
          |
    128 topology-selected sensory cells
          |
    fixed 1k MaleCNS recurrent graph
          |
    128 topology-selected readout cells
          |
    5 action logits

There are no Controller V2 risk/route heads in this policy.

## Training

`imitation_train.py` uses recurrent BPTT and class-balanced supervised cross
entropy against the expert action. Train/validation splitting is by whole
episode, so adjacent frames never leak across the split.

## Evaluation

`imitation_eval.py` loads only the saved neural checkpoint. It does not import
the expert planner and evaluates the MaleCNS policy by itself.
