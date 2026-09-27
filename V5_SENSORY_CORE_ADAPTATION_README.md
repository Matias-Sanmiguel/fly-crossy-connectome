# Crossy V5 Sensory/Core Adaptation

This is the architecture change triggered by the V4 critical-state surgery hard stop. It is **not another decoder audit**.

The selected `mlp64` motor decoder stays frozen. V5 changes the part that was still artificial and fixed in V4: the camera-to-`ol_sensory` interface.

## Phase 1 — sensory interface plasticity

V4 used a fixed random mapping from 48x24 RGB features into 4,114 `ol_sensory` neurons. V5 keeps that mapping as the baseline, but adds a small trainable calibration layer:

- one gain per sensory neuron;
- a rank-16 residual RGB -> `ol_sensory` projection;
- zero-initialized residual output, so the initial V5 policy is behaviorally identical to the selected V4 policy.

The decoder and measured MaleCNS core are frozen in this phase.

## Phase 2 — sensory/core boundary plasticity

If phase 1 does not reach the target, V5 also allows plasticity only at the immediate measured connectome boundary downstream of `ol_sensory`:

- edge gains for measured edges whose presynaptic neuron is `ol_sensory`;
- leak parameters for sensory neurons and their direct postsynaptic targets.

The rest of the measured MaleCNS remains frozen and the motor decoder remains frozen.

Training is stateful TBPTT. Each round recollects both teacher and actual student trajectories, strongly weights unsafe/unacceptable student states, and distills logits on student states that were already safe and acceptable to reduce catastrophic prefix regressions.

## Stopping rule

This script contains both intervention phases. It does not ask for another decoder audit afterward.

- Real success: 200 steps and score >= 80% of the teacher -> validate/integrate V5 for expo.
- Some real improvement but not yet target -> continue **the same V5 training** from `best.pt`.
- No improvement after both phases -> redesign the sensory topology and retrain the core as a new architecture.

## Run

```powershell
cd C:\Users\Nico\Documents\GitHub\fly-crossy-connectome
.\.venv\Scripts\Activate.ps1
cd python
pytest -q tests\test_v5_sensory_core_adaptation.py --basetemp .pytest-tmp
python -m fly_crossy.v5.sensory_core_adaptation --device cuda --smoke
python -m fly_crossy.v5.sensory_core_adaptation --device cuda
```

## Output

```text
runs\crossy-v5-sensory-core-adaptation\best.pt
runs\crossy-v5-sensory-core-adaptation\latest.pt
runs\crossy-v5-sensory-core-adaptation\report.json
```
