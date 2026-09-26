# Crossy V3 - fly-self-driving recipe

This is a deliberate reset of the learning architecture.

V2 froze the connectome and trained a small action readout. V3 follows the
closest successful visual-control MaleCNS project, suanmiao/fly-self-driving.

V3 uses:

- full traced MaleCNS graph, about 165k neurons and 25.56M measured edges;
- immutable adjacency;
- one trainable bounded gain per measured edge;
- one trainable bounded leak per neuron;
- recurrent state carried across decisions;
- four graph updates per decision;
- fixed random 64x32 grayscale pixel map into ol_sensory;
- fixed random vnc_motor readout into five Crossy action logits;
- behaviour cloning first;
- truncated BPTT, window 16, batch 4, Adam LR 0.04;
- three DAgger rounds, 500 updates each;
- the existing depth-4 Crossy planner is only the offline teacher.

Crossy-specific substitutions only:

- the street camera becomes the existing Crossy perspective neural camera;
- continuous steering becomes the five Crossy actions.

There is no DN+VPN decoder training, PPO, preference distillation, retina audit,
or cropped 4.6M-edge graph in this path.

## Prepare full MaleCNS

From the parent directory of fly-crossy-connectome:

```powershell
git clone https://github.com/MarkUnthank/flyhard
cd flyhard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python scripts\acquire_connectome.py
python scripts\prepare_graph.py
```

Expected files:

```text
flyhard\data\graph-traced-v1\graph.npz
flyhard\data\graph-traced-v1\nodes.feather
```

## Install

Extract this zip into the root of fly-crossy-connectome.

The script reuses your existing local V2 file:

```text
python\fly_crossy\v2\crossy_camera.py
```

Install PyArrow in the project venv if needed:

```powershell
pip install pyarrow
```

## Run

```powershell
cd python
pytest -q tests/test_v3_flyselfdriving_recipe.py --basetemp .pytest-tmp
python -m fly_crossy.v3.train_crossy --device cuda
```

Defaults mirror the successful recipe as closely as the discrete Crossy task
allows:

```text
60 expert demonstration episodes
15% safe action noise in demonstrations
16-decision TBPTT windows
batch 4
1200 behaviour-cloning updates
Adam LR 0.04
3 DAgger rounds
500 updates per DAgger round
30 training episodes driven by the fly per round
3 independent held-out sets of 20 worlds
```

Outputs:

```text
runs\crossy-v3-full-malecns\checkpoint.pt
runs\crossy-v3-full-malecns\report.json
```

## GPU

Try the RTX 5070 first. If the full 25.56M-edge backward pass runs out of VRAM,
do not crop the graph again. Move the same run unchanged to a larger remote GPU.
The point of V3 is to copy the architecture that already demonstrated strong
closed-loop visual control rather than invent another reduced variant.

## Source basis

Method adapted from the MIT-licensed repositories:

- suanmiao/fly-self-driving
- MarkUnthank/flyhard
