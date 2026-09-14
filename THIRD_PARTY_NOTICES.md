# Third-party materials

The custom license in `LICENSE` applies to original template work only.

- `public/data/brain-atlas/`: MaleCNS v1.0 curated soma locations, **CC BY 4.0**. Creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research. See the bundled `NOTICE.md` and `manifest.json` for source, transformations, filtering and hashes. https://male-cns.janelia.org/download/ and https://creativecommons.org/licenses/by/4.0/.
- `public/data/connectome/graph.json`: byte-identical reduced MaleCNS v1.0 graph reused from Mert Cobanov's **cobanov/flyjump**, commit **`c08c86bc18efd8125964b1d2ca4fc1df59700f30`**, https://github.com/cobanov/flyjump/tree/c08c86bc18efd8125964b1d2ca4fc1df59700f30. The MaleCNS data is **CC BY 4.0**; compatible application derivation material retains the **Cobanov Template Attribution License 1.0** already carried in this repository. Graph SHA-256: `2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719`. Its 80-cell selection was designed for FlyDino and may be suboptimal for Fly Crossy Connectome. No Dino game code or assets are included. Preserve the adjacent `NOTICE.md` and `manifest.json`.
- `public/data/flybody/`: derived Flybody anatomical body meshes, **Apache License 2.0**. Preserve its bundled `LICENSE` and `NOTICE.md`. Source: https://github.com/TuragaLab/flybody. The browser conversion originated in PinFly; the template removes its keyboard and motor animation.
- React / React DOM: MIT. Three.js: MIT. Vite and its React plugin: MIT. TypeScript: Apache-2.0. Type declarations and transitive dependencies retain their respective package licenses. Installed packages contain those texts; retain applicable notices when redistributing them. The custom template license does not replace their licenses.

No Refik Anadol Studio / Dataland videos, personal media, trained model weights,
private services, account identifiers or deployment credentials are included.
