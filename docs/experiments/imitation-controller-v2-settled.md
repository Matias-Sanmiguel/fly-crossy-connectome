# Imitation Controller V2 — Settled MaleCNS

The propagation audit showed that with the 128-sensory / 128-readout 1k graph,
0/128 readouts receive current sensory information after one internal update,
while 128/128 are reachable after exactly two updates.

V2 therefore holds the current ObservationV4 input on the sensory population
for two recurrent MaleCNS updates before sampling the readout.

Training uses balanced recurrent decision windows centered on FORWARD, LEFT,
RIGHT and WAIT. BACKWARD remains a legal output but is excluded from the core
selection metric because the current expert dataset contains only a few dozen
backward examples.

The exported browser policy contains `interfaceMode=population-settled` and
`internalSteps=2`. The expert planner remains offline-only and never chooses an
expo action.
