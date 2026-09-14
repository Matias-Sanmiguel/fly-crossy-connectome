# Reduced MaleCNS controller v1

## Scope

This experiment is a differentiable rate controller constrained by a small sourced connectivity graph. It is not a whole-brain simulation, a physiological neuron model, measured activity, or evidence that the selected biology is well suited to the game. Atlas locations are measured soma coordinates; the displayed values are simulated, dimensionless model state.

The 80-cell subset was selected for **FlyDino**, not Fly Crossy Connectome. Reusing it here is a predeclared engineering choice. The subset and its visual-to-descending selection bias may be suboptimal for this crossing task, so results apply only to this named reduced circuit.

## Pinned sources and license

- Dataset: **FlyEM MaleCNS v1.0, minimum confidence 0.5**.
- Dataset/project URL: <https://male-cns.janelia.org/download/>.
- Data creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research.
- Data license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, <https://creativecommons.org/licenses/by/4.0/>. No endorsement is implied.
- Reused reduced artifact and documented selection method: Mert Cobanov, [`cobanov/flyjump` commit `c08c86bc18efd8125964b1d2ca4fc1df59700f30`](https://github.com/cobanov/flyjump/tree/c08c86bc18efd8125964b1d2ca4fc1df59700f30), `public/data/connectome/graph.json` and its notice/manifest. Compatible application derivation material retains the **Cobanov Template Attribution License 1.0** already carried by this project.
- Byte-identical reused `graph.json` SHA-256: `2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719`.
- This project's normalized `ReducedGraphArtifact` SHA-256: `e7a2b3c1e2f4244b3fb01d838dd4ca4c677ab5eb5efdcfd9f59cb25171d70862`.

The upstream source-table SHA-256 pins are:

| MaleCNS v1.0 table | Source | SHA-256 |
| --- | --- | --- |
| minimum-confidence-0.5 annotations | [`body-annotations-male-cns-v1.0-minconf-0.5.feather`](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather) | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| minimum-confidence-0.5 directed edges | [`connectome-weights-male-cns-v1.0-minconf-0.5.feather`](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather) | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| neurotransmitters | [`body-neurotransmitters-male-cns-v1.0.feather`](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather) | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |

No Chromium Dino game code, sprites, policies, or unrelated application assets were copied.

## Selection and retained graph

The selection was determined from anatomy before game training:

1. For each of `LC4`, `LC11`, `LC9`, `LC15`, `LC16`, `LC17`, `LC21`, and `LPLC2`, rank cells by direct contact count onto descending neurons with soma coordinates and retain four. Ties are broken by body ID.
2. Take the union of the two strongest descending targets for each visual type, then fill to 16 descending targets by total contact rank.
3. Add 32 two-hop bridge cells ranked by the minimum of summed selected-visual input contacts and selected-descending output contacts.
4. Retain every measured directed edge among selected cells. The declared minimum edge threshold is one contact; recurrent and one-contact edges remain. No edges or body IDs are synthesized.

The result has **80 cells, 1,296 unique directed edges, 26,029 contacts, 32 input-role cells, and 16 readout-role cells**. All body IDs are checked against the atlas-visible ID set before construction. The builder also rejects a source hash mismatch, duplicate node IDs or output topology, invalid indices, non-finite values, missing metadata, and empty sensory/readout populations.

## Graph transformation and model

Let `c[j,i]` be the retained contact count from presynaptic cell `j` to postsynaptic cell `i`, and let the assumed transmitter sign `s[j]` be +1 for acetylcholine, -1 for GABA or glutamate, and 0 for unclear/modulatory annotations. These signs are modeling assumptions, not measurements of functional effect. The fixed recurrent adjacency is incoming-normalized:

```text
A[i,j] = c[j,i] s[j] / sum_k |c[k,i] s[k]|
```

A zero denominator yields zero drive. Topology and normalized recurrent edge weights are fixed buffers. For the 370-value `ObservationV1` encoding `o_t` and previous state `h_(t-1)`, one decision step is:

```text
r_t       = A h_(t-1)
candidate = tanh(S o_t + max(g, 0) r_t)
tau       = clamp(raw_tau, 1e-4, 1)
h_t       = h_(t-1) + tau (candidate - h_(t-1))
logits_t  = W_actor h_t + b_actor
value_t   = W_critic h_t + b_critic
```

`h` starts at zero and resets to zero at every episode boundary. PPO carries the state between decisions; its minibatch update treats each stored incoming state as fixed, a one-step truncated recurrent update.

Trainable parameters are the 370-to-80 sensory projection `S` without bias, recurrent gain `g`, time constant `raw_tau`, five-action actor readout, and scalar critic readout. That is 30,088 trainable scalars. Fixed elements are the 80 body IDs, 1,296-edge topology, normalized recurrent weights, action order, and observation encoding. Although the source artifact declares 16 descending readout-role cells for provenance and validation, v1 follows the reviewed implementation contract in using the full 80-value activity vector for the PPO actor and critic.

The browser exports actor inference only. It applies the same synchronous equation and preserves recurrent state until controller reset. Actor logits choose `forward`, `backward`, `left`, `right`, or `wait` by deterministic argmax. A shared Python/browser fixture verifies logits and node activity within absolute tolerance `1e-5`.

The bundled manifest's phrase **“Engineered 8-channel input injection”** describes the reused FlyDino source artifact and is retained as source provenance. It is not this controller's input interface. Fly Crossy encodes `ObservationV1` as 370 values and learns the 370-to-80 projection `S` shown above.

## Activity normalization

Raw state remains bounded by the leaky update when initialized in `[-1,1]`. The policy and recurrence use raw `h`. Only the atlas display mapping is normalized:

```text
display_activity = clamp((h + 1) / 2, 0, 1)
```

Those values are predicted model outputs keyed by the 80 validated MaleCNS body IDs. They are not firing rates, membrane voltages, calcium signals, or recordings.

## Known limitations

- The graph is a severe boundary truncation of MaleCNS and was chosen for a different game.
- The 370 engineered game inputs have no claimed biological receptive-field mapping; the learned sensory projection is artificial.
- Transmitter signs, normalization, tanh dynamics, one scalar gain, and one scalar time constant are simplified assumptions without physiological calibration.
- Missing external inputs and outputs at the subgraph boundary are omitted.
- PPO uses one-step truncated recurrent gradients rather than backpropagating through complete episodes.
- No comparison or positive score can establish biological fidelity or superiority of measured topology. Matched controls and held-out evaluation belong to the separately versioned evaluation task.

The machine-readable source, hashes, roles, and warning are in [`public/data/connectome/manifest.json`](../../public/data/connectome/manifest.json); the dataset notice is adjacent to it.
