# Crossy V4 Recovery DAgger V2

This patch is the follow-up to the V1 recovery report where a decoder reached `200/200` with only score `26`. That was a survival/stagnation exploit, not expert gameplay.

V2 keeps the trained MaleCNS checkpoint completely frozen and changes only the small motor readout experiment.

## What V2 changes

- **Real-progress success gate.** `200` steps alone no longer counts. The exact expo run must also reach at least `80%` of the held-out teacher score by default.
- **Stagnation detection.** Long no-score-progress streaks are terminated as `stagnation`. The cutoff is automatically raised if the teacher itself ever needs a longer streak.
- **Progress-aware preference loss.** The existing acceptable-set + pairwise-ranking + fatal-suppression planner loss is retained. An auxiliary term additionally prefers an immediately progressing action only when that action is also planner-acceptable.
- **No action-class balancing.** V1's rare-action class weighting is removed. Extra weight now goes to on-policy, recovery, unacceptable and unsafe states instead.
- **True on-policy recovery marking.** After the student makes an unacceptable/unsafe decision, following states are marked as recovery states for a short horizon.
- **Readout ablation.** Every stage compares:
  - `linear`: 708 -> 5
  - `mlp64`: residual 708 -> 64 -> 5 on top of the same initialized linear branch
- **Progress-first checkpoint selection.** A stalled low-score survivor cannot beat a model that makes more real progress.
- **Student-state semantic probe stays enabled.** This continues measuring whether the frozen MaleCNS representation degrades off-policy.

The measured MaleCNS remains unchanged: no connectome edge gain or neuron leak is retrained.

## Install

Extract the ZIP at the repository root:

```powershell
cd C:\Users\Nico\Documents\GitHub\fly-crossy-connectome

$zip = Get-ChildItem "$env:USERPROFILE\Downloads" -Filter "crossy-v4-recovery-dagger-v2*.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Expand-Archive `
    -Path $zip.FullName `
    -DestinationPath "C:\Users\Nico\Documents\GitHub\fly-crossy-connectome" `
    -Force

.\.venv\Scripts\Activate.ps1
cd python
```

## Validate

```powershell
pytest -q tests\test_v4_recovery_dagger_v2.py --basetemp .pytest-tmp
python -m fly_crossy.v4.recovery_dagger_v2 --device cuda --smoke
```

The smoke run uses a reduced 40-step setup and one recovery round.

## Full experiment

```powershell
python -m fly_crossy.v4.recovery_dagger_v2 --device cuda
```

Defaults:

- 12 teacher-anchor episodes
- 16 deliberately perturbed teacher episodes
- 16 on-policy episodes per round
- 5 recovery rounds
- 100 decoder epochs per architecture per stage
- linear vs residual MLP64 ablation
- planner depth 4
- success score = 80% of exact teacher score
- requested stagnation cutoff = 12 decisions, automatically increased if the teacher requires more

## Outputs

```text
runs/crossy-v4-recovery-dagger-v2/
  best_decoder.pt
  round-1-winner.pt
  ...
  report.json
```

The original V4 checkpoint remains untouched:

```text
runs/crossy-v4-expo-specialist/best.pt
```

## What to look for

The console prints, for both `linear` and `mlp64`:

```text
exactScore=...
steps=.../200
success=0|1
robust=.../21
TFacc=...
TFprogress=...
```

A result is only a real exact-seed success when:

```text
steps == 200
score >= minimumProgressScore
stalled == false
success == true
```

For the previous V1 failure mode (`200` steps, score `26`), V2 will mark the episode as stagnated or at minimum as `success=false`; it can no longer win checkpoint selection merely by surviving.

After the full run, send:

```text
runs/crossy-v4-recovery-dagger-v2/report.json
```
