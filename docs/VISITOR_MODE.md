# Visitor play screen

Open `http://127.0.0.1:5173/?play=1` while running `npm run dev`.
The existing `?expo=1` screen stays independent and can remain open on another display.

The visitor enters a name or nickname (1–32 printable Unicode characters),
then plays one human-controlled round. WASD / arrow keys move, Space waits,
and Escape pauses. There are also on-screen movement buttons. A new participant
starts from the name screen with a fresh server-generated seed.

The installation turn ends at the game's normal terminal condition, or after
900 simulation decisions (three unpaused minutes). This is a visitor-session
limit, not a change to simulation rules, rewards, world generation, or versions.
No connectome controller, neural activity, FlyBody, or biomechanics bridge is
loaded by this screen.

## Persistence and running

The Vite dev **and preview** servers provide `/api/visitors/` using Node's built-in
`node:sqlite` module; no extra npm dependency or database service is needed.
Use the project's Node >=22.18 requirement (verified locally with Node 24.21).
For the built site:

```powershell
npm run build
npm run preview
```

Open `http://127.0.0.1:4173/?play=1` for preview. Serving `dist/` from a static-only
server does not provide the ranking API. The page reports that limitation instead
of silently keeping a browser-only ranking.

The default database is `data/visitor-ranking.sqlite` under the repository root.
It is excluded from Git, including SQLite WAL/SHM sidecars. Names and scores are
installation-local, visible to everyone looking at the ranking. They are not
uploaded to a cloud service. Each completed round is a separate entry; repeated
names do not overwrite earlier rounds. The public list contains the best 10,
ordered by score descending and completion timestamp ascending, with the round
ID providing a deterministic final tie-breaker. All completed rounds remain in
the database even when outside the top 10.

To use a different database, set `VISITOR_DB_PATH` to an absolute file path before
starting the server. This is also how browser QA uses an isolated database.
Stop the local server before copying or archiving the database and any sidecars.
Do not delete the database to resolve a UI issue. No destructive reset endpoint
is exposed.

## Result validation

The server creates a UUID round ID and a seed using `crypto.randomUUID()`.
The world still uses its existing deterministic seeded generator; it does not
use `Math.random()`. A browser records every authoritative human-mode decision,
including idle waits, and sends the action sequence at the end of the round.
The server runs **the existing** `createGame` / `stepGame` functions on that seed,
rejects unfinished rounds and actions after death, and computes the score itself.
Client-supplied scores are never trusted. The visitor trace has a 900-action cap,
and the browser refuses to submit if it missed a simulation step.

A completed round stores a trace SHA-256. Repeating the same submission returns
the same result; submitting a different trace for that ID is rejected. If saving
fails, the result screen preserves the trace and offers a retry. Returning to the
name screen without saving is an explicit choice. Reloading/closing a tab during
a live round abandons it; incomplete rounds do not enter the ranking.

The API uses parameterized SQL, bounded request bodies, same-origin checks on
browser writes, and a 60-write-per-minute installation limit. It runs with the
existing loopback-only dev/preview default. This is a trusted local event kiosk,
not an authenticated public tournament or a proof of real-time keyboard input.
Do not expose the Vite server publicly. Gameplay-version changes must update the
ranking ruleset in `server/visitorStore.ts` to avoid combining incompatible scores.

## Verification

```powershell
node --experimental-strip-types --test tests/visitor-mode.test.mjs tests/expo-mode.test.mjs
npm test
npm run check:assets
npm run build
```

Focused tests cover name/action validation, independent seeds, authoritative
score calculation, unfinished/post-terminal rejection, the visitor time limit,
save idempotency, ranking order/top-10 bounds, persistence after reopening,
HTTP body/origin handling, and isolation from expo/neural controllers.
