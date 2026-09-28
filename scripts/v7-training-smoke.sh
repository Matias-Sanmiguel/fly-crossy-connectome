#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(mktemp -d)
repo_mount="$repo_root"
run_mount="$run_dir"

if [[ "${MSYSTEM:-}" == MINGW* ]]; then
  repo_mount=$(cygpath -m "$repo_root")
  run_mount=$(cygpath -m "$run_dir")
  export MSYS2_ARG_CONV_EXCL="*"
fi
chmod 0777 "$run_dir"
started=$(date +%s)
trap 'rm -rf -- "$run_dir"' EXIT

release_before=$(find "$repo_root/release" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)

docker run --rm \
  -v "$repo_root:/workspace:ro" \
  -v "$run_mount:/run-v7" \
  -w /workspace/python \
  -e PYTHONPATH=/workspace/python \
  stabilize-origin-main-simulation \
  python -m fly_crossy.v7.curriculum \
    --profile smoke \
    --out /run-v7 \
    --device cpu \
    --time-budget-seconds 180

docker run --rm \
  -v "$run_mount:/run-v7:ro" \
  stabilize-origin-main-simulation \
  python -c '
import json, math, torch
from pathlib import Path
root = Path("/run-v7")
report = json.loads((root / "report.json").read_text())
assert report["contractOnly"] is True
assert report["decision"]["nextAction"] != "promote"
assert report["rounds"]
for section in ("teacherLoss", "studentLoss"):
    assert all(math.isfinite(float(v)) for v in report["rounds"][0][section].values())
best = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
latest = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
assert best["student"]
assert latest["teacher"] and latest["student"]
'

release_after=$(find "$repo_root/release" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
test "$release_before" = "$release_after"

elapsed=$(($(date +%s) - started))
echo "V7 smoke report: $run_dir/report.json"
echo "Elapsed: ${elapsed}s"
