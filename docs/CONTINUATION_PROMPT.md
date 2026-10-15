# Prompt to continue the project

You are continuing `fly-crossy-connectome`. Read `README.md`,
`docs/ACTIVE_CONTEXT.md`, `docs/HANDOFF.md` and `docs/TRAINING_ARTIFACTS.md` before
making substantive changes. Preserve the existing architecture and verify the
part of the project the user asks to change.

The current installation has two separate browser modes:

- `?play=1`: human-only game, registration and local SQLite ranking. Run with
  Vite dev/preview; preserve the database and existing participant results.
- `?expo=1`: fixed Full MaleCNS actions and synchronized exported model activity.
  Preserve the three `public/assets/expo/full-malecns-*` files and their hashes.

The user confirms a teammate has the full-neuron training bundle on their
machine and keeps it outside Git because it is large. Do not assume it was
never trained, recreate it, or restart V7 simply because checkpoints are absent
from this checkout. Inspect external artifacts only when they are provided
and their analysis is requested. Distinguish displayed exports, continuous
inference and generalization evidence.

Normal `/` live autoplay and the Python physical station currently use the
released 80-neuron compatibility controller. V7 and the versioned
cached-frontier scripts are separate research paths; the latter now live under
`scripts/experiments/cached_frontier/`. Historical notes are under
`docs/experiments/legacy/`. Do not merge versions or change serialized classes
without checking historical checkpoint compatibility.

Generated `runs/` files stay local and ignored. Historical report summaries,
hashes, byte sizes and the original source revision are recorded in
`docs/experiments/run-index.json`. Preserve immutable `release/` evidence,
asset provenance, licenses and third-party notices. Do not add agent coauthor
trailers. Do not rewrite shared history to remove old blobs without a separate
explicit request.

Known integration limits: the web Docker build omits `server/`, Nginx does not
host the visitor-ranking API, and expo labels still say “in real time” even
though that mode consumes recorded activity. These are separate changes from
the repository cleanup; do not silently broaden a task to include them.

For frontend changes, run `npm test`, `npm run check:assets` and `npm run build`.
For Python/runtime changes, run relevant tests in the declared Python
environment and check physics authority and browser/Python parity. In
biomechanical mode, only confirmed physical contact may cause directional
game movement. Do not replace contact confirmation with a visual animation.

Report what changed, what was freshly verified, and any remaining constraints.
Keep research completion claims separate from the human/replay installation.
