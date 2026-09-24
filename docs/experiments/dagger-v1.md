# DAgger V1

The expert planner remains development-only.

For each DAgger state:

1. the settled MaleCNS receives ObservationV4 and chooses the action that is
   actually executed;
2. the offline depth-4 planner labels that same pre-action state;
3. the observation plus expert label is stored;
4. the next state is produced by the MaleCNS action, not the planner action.

This exposes the student to the distribution created by its own mistakes while
keeping the target action expert-labelled.

The output dataset aggregates the original expert trajectories and the new
student-visited trajectories. It preserves complete episode boundaries for
recurrent training. Additional arrays record student actions, source, and
student/expert disagreement for diagnostics; the imitation trainer continues to
use only observations and expert action labels.

No planner state, planner value, or planner action is exported to browser
inference.
