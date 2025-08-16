## Running Simulations via run_simulation.py

This guide explains how to launch a single simulation run using the unified CLI entry point.

### Arguments

- --scenario: Required. One of: highway, merge, intersection, roundabout, racetrack
- --composition: Required. One of: selfish, cooperative, mixed
- --render: Optional flag. If set, enables on-screen visualization (render_mode=human)
- --output-csv: Optional path. Appends one row of run-level metrics to this CSV
- --output-steps-csv: Optional path. Saves per-step metrics for the run
- --config-overrides: Optional JSON string to override base config.yaml values

### Examples

```bash
# Highway, selfish, with rendering and outputs
python run_simulation.py --scenario highway --composition selfish --render \
  --output-csv results/results.csv --output-steps-csv results/highway_selfish_steps.csv

# Merge, mixed, headless run
python run_simulation.py --scenario merge --composition mixed \
  --output-csv results/results.csv

# Roundabout, cooperative, override duration
python run_simulation.py --scenario roundabout --composition cooperative \
  --config-overrides '{"simulation": {"duration_steps": 200}}'
```

### Notes

- The old tkinter GUI has been removed; use this CLI for all runs.
- Per-step CSV is useful for deep analysis; the aggregate CSV contains one row per run.


