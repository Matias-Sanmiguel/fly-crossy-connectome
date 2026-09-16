# Fly Crossy Connectome — agent instructions

## Read first

Before making substantive changes, read:

- `docs/ACTIVE_CONTEXT.md`
- the relevant source files and tests for the requested area

Do not infer project intent only from filenames or existing implementation.

`docs/ACTIVE_CONTEXT.md` records current decisions, verified milestones,
known temporary behavior, and the active development phase.

## Project intent

Fly Crossy Connectome is an interactive installation in which a simulated
fruit-fly connectome controller ultimately drives FlyBody in MuJoCo, physically
presses W/A/S/D keyboard keys, and only confirmed physical key contact may
advance the authoritative Crossy-style game.

The browser game is also used independently for development, training,
evaluation, and visual QA.

## Critical architectural rules

- Keep game simulation/state separate from rendering.
- Visual changes must not silently change gameplay.
- Do not use `Math.random()` for authoritative or reproducible visual choices.
- Seeded generation must remain deterministic.
- Browser and Python environment behavior must remain in parity when gameplay
  or reward rules are changed.
- In biomechanics mode, requested neural actions must never directly move the
  game. Only the physical action result may do so.
- Do not bypass the FlyBody -> physical key -> contact confirmation chain.
- Do not claim neural/biological behavior that the implementation does not
  actually provide.
- Preserve provenance, hashes, manifests, and licensing for external assets.

## Development rules

- Inspect existing tests before changing behavior.
- Add or update tests for behavioral changes.
- Run focused tests first, then the broader suite/build.
- Do not train larger neural populations until the environment/gameplay freeze
  described in `docs/ACTIVE_CONTEXT.md`.
- Do not rewrite working architecture merely for stylistic reasons.
- Prefer incremental changes that preserve reproducibility.

## Validation

Frontend / game:

```powershell
npm test
npm run check:assets
npm run build
```

Python:

```powershell
cd python
python -m pytest -q --maxfail=1 --basetemp=.pytest-tmp
```

Use narrower relevant tests during implementation before running full suites.

## Asset direction

The game uses a low-poly visual language based primarily on Kenney assets.

Selected packs currently include:

- Car Kit
- City Kit (Roads)
- Train Kit
- Mini Forest

The fly itself remains a custom procedural model unless explicitly changed.