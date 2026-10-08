# Expo neural activity

Expo uses the existing Full MaleCNS action replay, with corresponding recurrent
model states rather than the unrelated released 80-neuron policy. This is
simulation output on measured anatomy, **not a recording of a living fly**.
The interface labels, soma coordinates and gameplay are unchanged.
Expo uses a display-only contrast profile: a fixed gain of 12 and a brightness
floor of 0.5 for nonzero signals; zero stays blue. The standard brain view keeps
its previous transfer curve and marker sizes. Only Expo's highlighted markers
use a 2.6-pixel diameter (times device pixel ratio); unhighlighted anatomy stays
at the previous size. There is no independent blink clock or random pulse.

Regenerate from the referenced local checkpoint and its committed run state:

```powershell
.venv/Scripts/python.exe scripts/export-expo-activity.py --device cuda
```

This performs inference only. It restores the original committed float32 neural
anchors used by the cached-frontier training workflow, renders its committed
observation snapshots with the existing RGB camera, and runs the same four neural
updates per decision. Each snapshot must equal the game state replayed from the
recorded actions; the exporter fails if simulation and trace diverge.
The committed decoder must match the checkpoint exactly; all committed actions
and all reconstructed decoder decisions must match the existing action replay.
Nothing is exported if any check fails. A continuous rollout without those
anchors can differ at low-margin decisions (observed at step 617 on GPU and 385
on CPU), so it must not silently replace this historical replay.

Only graph body IDs with measured, visible somata are eligible. The exported
value is signed-state magnitude with fixed regional display gains: optic=1,
central=16, descending=16, clipped to [0,1] and quantized to uint8 / 255.
These fixed gains reveal weaker genuine central signals without frame-wise
renormalization; zero remains zero. Magnitudes across regions are therefore
not directly comparable in the display.
For performance, retain at most 1,536 nonzero signals per frame: 512 per measured
optic/central/descending region. Half of each quota prioritizes actual changes
in quantized magnitude since the previous sample; the remaining slots take the
strongest nonzero values. Ties use ascending body ID. Missing signals do not
get fabricated to fill a quota. A soma returning to blue can also mean it fell
outside the display selection, not that a biological neuron stopped firing.
The compact pool is the union of those selections, not invented extra somata.
The full model still runs all 165,122 neurons. Zero stays zero. This
visualizes state magnitude, not spikes, firing rate or a biological measurement.
There is no per-frame rescaling, interpolation, random activation or added soma.
The renderer reuses its measured-ID index and typed activity buffer instead
of allocating a new Map per decision. Changed selections and real magnitude
variations drive the visual on/off changes; no artificial spike train is added.

The gzip file contains frame-major uint8 magnitudes, with body-ID ordering in
the JSON manifest. That manifest records checkpoint/run-state hashes, the exact
action-sequence hash, and the decompressed-data hash. The browser validates the
seed, sequence, byte count, hashes and visible IDs before starting the replay.
One decision consumes one frame; pause consumes none, restart rewinds both,
and aborted decisions do not advance the cursor. Corrupt/mismatched data fail
closed rather than silently showing activity from another model.
