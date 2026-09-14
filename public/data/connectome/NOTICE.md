# MaleCNS v1.0 circuit subset

Data creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. Dataset/project: https://male-cns.janelia.org/download/

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. No endorsement of this experiment is implied.

Derived from the v1.0 minimum-confidence-0.5 annotation and connectivity tables, and v1.0 neurotransmitter table. Exact URLs and SHA-256 are documented in the repository protocol and manifest.

Changes: deterministic selection of 80 cells and all 1,296 measured internal directed edges; retention of original body IDs, soma coordinates, type and transmitter annotations; creation of graph indices and engineered input/output roles. Source edge contact counts remain unchanged. The runtime normalizes incoming contact counts and applies assumed transmitter signs and leaky tanh dynamics. These transformations are not biological measurements.

This byte-identical `graph.json` and its documented selection method were reused from Mert Cobanov's `cobanov/flyjump`, commit `c08c86bc18efd8125964b1d2ca4fc1df59700f30`: https://github.com/cobanov/flyjump/tree/c08c86bc18efd8125964b1d2ca4fc1df59700f30. MaleCNS graph data remains CC BY 4.0. Compatible application derivation material retains the Cobanov Template Attribution License 1.0 already carried by this project. No Dino game code or assets are included.

Important scope limit: these 80 cells were selected for FlyDino, not for Fly Crossy Connectome. The subset may be suboptimal for this task and must not be treated as a representative MaleCNS sample.

The graph represents a small selected circuit, not a complete brain. Its activity is simulated and dimensionless. Input encoding and action readout are artificial.

Original subset derivation: the pinned upstream [`scripts/build-connectome.py`](https://github.com/cobanov/flyjump/blob/c08c86bc18efd8125964b1d2ca4fc1df59700f30/scripts/build-connectome.py). That script checks the source-table hashes before extraction; use it from the pinned upstream commit, not as a claimed local file. This project's derivation, equations, boundary assumptions, hashes, and validation are documented at `docs/experiments/reduced-connectome-v1.md`.
