# Fly Connectome Template

A source-available starting point for your own fly-connectome experiments.
Real anatomical assets, an environment slot, and a model-output viewer with
explicit provenance. **No trained model, RL policy, neural simulator, game,
art feed, private backend or hosting account is bundled.**

**License: attribution is required in both your web UI and repository README.**
Read [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md) before reusing.
This custom license is not advertised as OSI-approved. Your own model weights,
data and modifications do not have to be published.

## Start your experiment

1. Click **Use this template → Create a new repository** on GitHub (or [generate a repository](https://github.com/new?template_name=fly-connectome-template&template_owner=cobanov)).
2. Clone your new repository. Install Node.js 22.18+ (or a compatible newer LTS).
3. Run:

   ```sh
   npm ci
   npm run dev
   ```

4. Open the URL printed by Vite. The initial view contains real anatomy and a generic moving-spot stimulus. It generates **no neural activity**.
5. Click **Load synthetic example**, then **Play**, to exercise the output pipeline with clearly labeled demonstration values. Use **Load model JSON** for your own exported data. Files are read locally in the browser, not uploaded to a server.

## What you can replace

- `src/components/Environment.tsx`: your game, video, sensory scene or environment UI. The starter spot is an example, not a biological visual encoder.
- `src/components/BrainScene.tsx`: real MaleCNS soma rendering. Takes `Atlas` plus `ActivityFrame | null`; values map by **body ID**, never list order or spatial proximity.
- `src/components/FlyScene.tsx`: anatomical body surface. Add motor decoding or a real physics adapter here; the starter has no controller or leg animation.
- `src/lib/replay.ts`: documented model-output contract and validator. See [model integration](docs/MODEL-INTEGRATION.md).
- `src/App.tsx`: experiment controls, clock and example replay. Replace replay with your own inference adapter, retaining source labels and time alignment.
- `src/style.css`: layout and styling. The console defaults to environment left, brain upper right, body lower right; it stacks on mobile.
- `src/components/Attribution.tsx`: mandatory linked template credit. Restyle or relocate it as allowed in the license; do not remove it. Preserve the README credit below too.

The code is plain **React + TypeScript + Three.js + Vite**. There is no Next.js,
Sites registration, database, account ID, domain-specific deploy command or
required cloud service. Assets resolve relative to Vite's base URL, so the
built app can also be hosted under a subpath.

## Scientific scope

- **Anatomy:** official adult male **MaleCNS v1.0** annotations, not female FlyWire. 140,024 curated `somaLocation` records are bundled; 124,289 classified optic, central and descending somata are drawn. VNC-associated and unclassified cells are excluded from the brain view. Traced cells without a soma location are omitted, never replaced with synthetic points.
- **Geometry:** native isotropic 8 nm coordinates, a rigid `(x,y,z) → (x,-y,-z)` rotation, centering and one uniform scale. No per-axis stretching. “XY view” pauses the slow orbit and resets the projection. Point sizes are display markers, not cell diameters.
- **Coverage:** soma positions are not the full connectome graph, neurite branches or a neuropil surface. The class filter is not a complete anatomical brain segmentation.
- **Activity:** none by default. The synthetic example is authored test data. Imported `predicted`/`measured` labels are declarations by the file author, not validation of scientific truth. Values must be normalized to [0,1], with a written normalization rule. Missing entries mean zero for that frame. Rendering holds the last supplied frame; it adds no random firing, interpolation or temporal decay.
- **Body:** Flybody anatomical mesh, not MuJoCo simulation. Training and inference are separate from rendering.

Source, hashes, filters and transforms: [atlas manifest](public/data/brain-atlas/manifest.json).
Credits: [third-party notices](THIRD_PARTY_NOTICES.md), [MaleCNS notice](public/data/brain-atlas/NOTICE.md), [Flybody notice](public/data/flybody/NOTICE.md).

### Reproduce the anatomy export

```sh
curl -L 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather' -o /tmp/malecns-v1.feather
uv run --with pyarrow python scripts/build-brain-atlas.py /tmp/malecns-v1.feather --check
```

This audits the pinned source hash and every exported coordinate/body ID without
writing. Omit `--check` to reproduce the export. Source data retains CC BY 4.0;
the template's custom license does not add restrictions to it.

## Validation and deployment

```sh
npm test
npm run check:assets
npm run build
npm run preview
```

GitHub Actions runs the tests, anatomical checksums and production build.
Deploy the generated `dist/` directory to any static host. On Cloudflare Pages,
use build command `npm run build` and output directory `dist`. Create your own
hosting project; no deployment credentials are included. GitHub Pages or a
plain HTTP server can serve the same build. GPU training cannot run on a
static host; replay files or a separately operated inference service provide
outputs to the frontend.

## Required attribution

Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template) by [Mert Cobanov](https://github.com/cobanov).

This original template is provided under the [Cobanov Template Attribution
License 1.0](LICENSE). Derived versions must keep the linked credit in their
web UI and repository README and identify that they were modified. Third-party
anatomical assets and dependencies retain their own licenses. A public GitHub
repository does not make this code public domain or remove its attribution
conditions.
