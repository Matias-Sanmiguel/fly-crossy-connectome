# Historical Full MaleCNS cached-frontier experiments

These versioned scripts were moved from the repository root without changing
their training logic. Keep every version: historical checkpoints and saved
traces may refer to its classes and behavior. The expo exporter still imports
`TraceStep` from `repair_full_malecns_cached_frontier_v12_10` using this directory
on its module search path.

Run from the repository root, for example:

```sh
python scripts/experiments/cached_frontier/repair_full_malecns_cached_frontier_v12_10.py --help
```

Use the original Python environment and external checkpoints for the selected
experiment. These are research tools, not prerequisites for playing the game
or displaying the already exported expo. Reports/checkpoints keep their
original locations under the ignored `runs/` directory.

Historical notes are in [docs/experiments/legacy](../../../docs/experiments/legacy/).
See [TRAINING_ARTIFACTS.md](../../../docs/TRAINING_ARTIFACTS.md) for external files.
