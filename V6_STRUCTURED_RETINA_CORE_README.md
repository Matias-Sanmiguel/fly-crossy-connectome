# Crossy V6 Structured Retina + Core Curriculum

V6 is the architecture change triggered by the V5 hard stop. It is not another audit and it does not continue tuning the old random/global sensory adapter.

## What changes

V4/V5 started from an artificial random RGB-to-`ol_sensory` routing. V6 removes that routing entirely.

The new input path is:

```text
48x24 RGB
  -> shared local 3x3 retina filters
  -> deterministic spatial routing
  -> 4,114 ol_sensory neurons
  -> measured MaleCNS
  -> 708 vnc_motor neurons
  -> frozen MLP64 motor interface
```

The retina has eight local feature maps initialized as luminance, color-opponent, local-contrast and local-edge filters. The filters are trainable, but every sensory neuron remains tied to a local image position. There is no global pixel mixing and no random pixel assignment.

## One training, three phases

V6 contains the full curriculum in a single run:

1. `retina-only`: train only the structured retina.
2. `retina-hop1`: also allow measured edges leaving `ol_sensory` and sensory/first-hop leaks to adapt.
3. `retina-hop2`: expand plasticity one additional measured-connectome hop.

The selected MLP64 decoder stays frozen in all phases.

Every round starts from the current best checkpoint. Rejected candidates do not accumulate.

## Decision rule

V5 ended at score 41. V6 is only considered architecturally promising if the exact expo seed reaches at least score 60. The final report therefore emits one of:

```text
validate-and-integrate-v6-for-expo
continue-v6-structured-retina-training
reject-v6-structured-retina-topology-and-redesign
```

The real expo success criterion remains 200 steps and score >= 80% of the teacher.

## Run

```powershell
cd C:\Users\Nico\Documents\GitHub\fly-crossy-connectome
.\.venv\Scripts\Activate.ps1
cd python

pytest -q tests\test_v6_structured_retina_core.py --basetemp .pytest-tmp
python -m fly_crossy.v6.structured_retina_core --device cuda --smoke
python -m fly_crossy.v6.structured_retina_core --device cuda
```

## Output

```text
runs\crossy-v6-structured-retina-core\best.pt
runs\crossy-v6-structured-retina-core\latest.pt
runs\crossy-v6-structured-retina-core\report.json
```
