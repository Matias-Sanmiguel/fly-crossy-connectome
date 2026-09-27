# V4 Exact-Seed Branch Curriculum

This experiment is intentionally specialized to the deterministic expo seed.

It does **not** retrain the 25.5M measured MaleCNS edge gains/leaks. It loads:

- `runs/crossy-v4-expo-specialist/best.pt` for the frozen MaleCNS;
- `runs/crossy-v4-recovery-dagger-v2/best_decoder.pt` as the starting motor readout.

The exact teacher trajectory is replayed once and the recurrent MaleCNS state is snapshotted at every decision. The curriculum then advances through teacher-prefix windows (default `0:50`, `50:100`, `100:150`, `150:200`). At each root it executes up to two **immediate-safe deviations**, preferring planner-unacceptable deviations, and then lets the planner recover for 12 decisions. Crucially, the branched neural rollout starts from the actual recurrent MaleCNS state at that teacher prefix rather than from a zero state.

The MLP64 decoder is fine-tuned cumulatively after each deeper window. Model selection keeps the V2 real-progress rule: surviving without progress cannot win. Default success requires 200 decisions and at least 80% of the exact teacher score.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
cd python
pytest -q tests\test_v4_exact_seed_branch_curriculum.py --basetemp .pytest-tmp
python -m fly_crossy.v4.exact_seed_branch_curriculum --device cuda --smoke
python -m fly_crossy.v4.exact_seed_branch_curriculum --device cuda
```

Outputs:

```text
runs/crossy-v4-exact-seed-branch-curriculum/best_decoder.pt
runs/crossy-v4-exact-seed-branch-curriculum/stage-1.pt
...
runs/crossy-v4-exact-seed-branch-curriculum/report.json
```

The primary result is the exact seed score/steps/success after every branch-curriculum stage. Robust 21-variant evaluation is retained as a secondary diagnostic only; the experiment's main goal is reliable expo specialization on `crossy-v4-expo:0006`.
