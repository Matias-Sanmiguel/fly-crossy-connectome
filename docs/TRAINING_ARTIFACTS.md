# External training artifacts

The team's full-neuron training artifacts are kept on a teammate's machine
because of their size. Their absence from Git is intentional. It does not mean
that training was never performed, and the cleanup does not retrain, replace or
evaluate those checkpoints.

## What each mode needs

| Mode | Artifacts needed |
| --- | --- |
| Human game, `?play=1` | Bundled scene assets and local Vite ranking API; no training checkpoint |
| Expo, `?expo=1` | The three committed `public/assets/expo/full-malecns-*` exports |
| Released live autoplay / physical station | Reviewed artifacts already in `release/`, plus the Python environment for physics |
| Regenerating the expo or continuing training | The original external checkpoint, training state, graph and referenced source artifacts |

The expo exports are preserved byte-for-byte. Its checkpoint SHA-256 remains in
`public/assets/expo/full-malecns-activity.json`. Displaying those exports does
not require loading the original checkpoint. See [expo-neural-replay.md](expo-neural-replay.md)
for the recorded-state reconstruction and display contract.

## Sharing the full training bundle

Transfer the original bundle through shared storage, an external drive or a
separate artifact store. Preserve run-directory names and relative layout
under `runs/`. Include the selected checkpoint (for the current export,
`crossy-full-readout-cached-frontier-v12-10-0006-1000/best_FROZEN_1000.pt`), its
matching `run_state.pt`, and the source checkpoints, decoder and graph required
by that experiment. A single `best.pt` is not necessarily a resumable bundle.

Include a small manifest with relative filenames, sizes, SHA-256 hashes,
training configuration, code revision and Python/package versions. Do not
upload participant databases with a training bundle. Keep the original bundle
until a transferred copy has been verified.

The exporter currently reads historical absolute Windows paths from
`sourceCheckpoint`, and source-checkpoint/decoder paths inside the checkpoint.
Copying one file into `runs/` does not automatically make those paths portable.
Use the original environment or explicitly map those paths in a separate
integration change. Do not silently substitute another model or edit published
provenance just to make an export succeed.

`runs/`, `artifacts/`, new `reports/` output, local backups and checkpoint
extensions are ignored. Existing reviewed checkpoints in `release/` remain
tracked and distributable. Promote new artifacts into `release/` only with
their evaluation evidence and manifest, not as incidental cleanup.

## Historical run reports

[experiments/run-index.json](experiments/run-index.json) records the 61 generated
files removed from Git tracking in this cleanup, their original Git-byte sizes,
SHA-256 hashes, selected reported metrics, and the source commit. These are
historical report claims, not a fresh validation of the external model.
The local files are left in `runs/`; previously published releases and curated
audit evidence in `reports/` are retained.

The original files can also be read from Git history. For example:

```sh
git show 6a7c00e:runs/crossy-full-readout-cached-frontier-v12-10-0006-1000/report.json
```

Use a full clone if a shallow clone does not include that commit. Hashes in the
index refer to Git blob bytes; a local CRLF/LF conversion can change the
filesystem hash without changing JSON content.

This cleanup does not rewrite shared Git history. It reduces the tracked
working tree and future build inputs; old blobs still exist in Git history.
Before merging this branch on another machine, back up its local `runs/`
directory: Git may remove unchanged formerly tracked files during checkout or
merge. Preserve modified run outputs rather than force-resetting them.
