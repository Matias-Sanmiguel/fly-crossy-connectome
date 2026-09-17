# MaleCNS v1.0 1000-cell capacity subset

Data creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research.
Dataset/project: https://male-cns.janelia.org/download/

License: Creative Commons Attribution 4.0 International.
No endorsement of this experiment is implied.

This artifact is derived from the verified MaleCNS v1.0 minimum-confidence-0.5
annotation and directed connectivity tables and the v1.0 neurotransmitter table.
Exact source URLs and SHA-256 values are recorded in `manifest.json`.

The historical 80-cell circuit is retained in full. The additional 920 cells are
selected deterministically using measured anatomical connectivity to that frozen
core only. Fly Crossy rewards, scores, policies, training seeds, evaluation seeds
and game outcomes are not used for selection.

All measured directed internal source-target connections among the selected 1000
cells are retained, with duplicate rows aggregated by contact count. The original
32 engineered input interfaces and 16 engineered readout interfaces are preserved;
new cells receive no game-specific input/output role.

The runtime's signed normalization and recurrent leaky-tanh dynamics are modeling
assumptions, not biological measurements. This 1000-cell circuit is a bounded
capacity experiment, not a complete or representative fly brain.
