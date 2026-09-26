# DAgger Training V1

Round 1 showed a 40.46% student/expert disagreement rate on states visited by
the settled MaleCNS itself.

The largest errors were not represented by the original expert-only
validation distribution:

- expert LEFT agreement on student states: about 30%;
- expert RIGHT agreement: about 33%;
- expert BACKWARD: 287 labels, zero student BACKWARD actions;
- student WAIT was strongly over-produced relative to the expert.

The fine-tuner therefore does not return to uniform class-balanced replay.
Uniform replay was useful to escape FORWARD collapse, but it distorted the
action prior, especially WAIT.

Hybrid replay uses:

- 50% anchors sampled naturally from the full aggregated dataset;
- 25% anchors sampled from DAgger states the previous student disagreed on;
- 25% anchors sampled across supported action classes.

BACKWARD automatically joins balanced replay once it has at least 64 training
examples. Round 1 has enough BACKWARD examples to cross that threshold.

The model is initialized from the exact checkpoint that generated the DAgger
rollouts. Architecture and browser inference remain unchanged:
ObservationV4 -> 128 sensory -> 1k MaleCNS with two internal updates ->
128 readout -> five actions.

The planner remains offline-only.
