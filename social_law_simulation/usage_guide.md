## Running Simulations via run_simulation.py

This guide explains how to launch simulations using the unified CLI entry point.

### Arguments

- --config: Optional. Path to unified `config.yaml` (defaults to project `config.yaml`).
- --scenario: Optional. One of: highway, merge, intersection, roundabout, racetrack. If omitted, uses enabled scenarios from `config.scenarios`.
- --composition: Optional. One of: selfish, cooperative, mixed, all. If omitted, uses compositions listed under the scenario in `config.scenarios`.
- --runs: **NEW** Optional integer. Number of runs per scenario-composition combination (overrides config).
- --seed: **NEW** Optional integer. Base random seed for reproducibility (default: 0).
- --render: Optional flag. If set, enables on-screen visualization (render_mode=human).
- --output-csv: Optional path. Appends one row of run-level metrics to this CSV.
- --output-steps-csv: Optional path. Saves per-step metrics for a single run.
- --output-results: **NEW** Optional path. Saves aggregated results CSV (one file per invocation with statistical summaries).
- --save-plots: **NEW** Optional flag. Generates and saves enhanced comparison plots with delta annotations.
- --config-overrides: Optional JSON string to override values from the config (deep-merge).
- --logs-dir: Optional. Directory to write log files. Defaults to `output.logs_dir` in config ("logs").

### Examples (on-screen visualization only)

```bash
# Highway
python run_simulation.py --scenario highway --composition selfish --render
python run_simulation.py --scenario highway --composition cooperative --render
python run_simulation.py --scenario highway --composition mixed --render

# Merge
python run_simulation.py --scenario merge --composition selfish --render
python run_simulation.py --scenario merge --composition cooperative --render
python run_simulation.py --scenario merge --composition mixed --render

# Intersection
python run_simulation.py --scenario intersection --composition selfish --render
python run_simulation.py --scenario intersection --composition cooperative --render
python run_simulation.py --scenario intersection --composition mixed --render

# Roundabout
python run_simulation.py --scenario roundabout --composition selfish --render
python run_simulation.py --scenario roundabout --composition cooperative --render
python run_simulation.py --scenario roundabout --composition mixed --render

# Racetrack
python run_simulation.py --scenario racetrack --composition selfish --render
python run_simulation.py --scenario racetrack --composition cooperative --render
python run_simulation.py --scenario racetrack --composition mixed --render

# Config-driven batch (runs all enabled scenarios/compositions from config)
python run_simulation.py --config config.yaml --render
```

### Examples (with enhanced results pipeline)

```bash
# Full batch with aggregated results and enhanced plots
python run_simulation.py --config config.yaml --output-results results/aggregated.csv --save-plots

# Multiple runs for statistical robustness with plots (all compositions)
python run_simulation.py --scenario highway --composition all \
  --runs 10 --seed 42 --output-results results/highway_stats.csv --save-plots

# Single scenario with detailed outputs
python run_simulation.py --scenario merge --composition cooperative \
  --output-results results/merge_coop.csv --save-plots \
  --output-steps-csv results/merge_coop_steps.csv

# Quick experiment with config overrides and plots
python run_simulation.py --scenario roundabout \
  --config-overrides '{"environment": {"roundabout": {"vehicles_count": 20}}}' \
  --output-results results/roundabout_light.csv --save-plots
```

### Multi-Run Execution and Reproducibility

The new `--runs` and `--seed` options enable robust statistical analysis:

```bash
# Run 20 iterations with consistent seeding for reproducibility
python run_simulation.py --scenario merge --composition cooperative \
  --runs 20 --seed 123 --output-results results/merge_robust.csv

# Compare all compositions with 15 runs each
python run_simulation.py --scenario roundabout --composition all \
  --runs 15 --seed 456 --output-results results/roundabout_comparison.csv --save-plots
```

**Seeding behavior**:
- `--seed 42` sets the base seed
- Each run gets `seed + run_index` (42, 43, 44, ...)
- Ensures reproducible results across invocations
- Seeds are applied to Python's `random`, `numpy.random`, and environment resets

**Composition shortcuts**:
- `--composition all` runs selfish, cooperative, and mixed for the given scenario
- Saves time when comparing all agent behaviors

### Plot Features

When using `--save-plots`, the generated visualizations include:

- **Sample size annotations**: Each bar shows N=X indicating the number of simulation runs
- **Delta annotations**: Percentage changes vs. selfish baseline (e.g., "+15.2%" for cooperative improvements)
- **Color-coded deltas**: Green for positive changes (speed ↑, merge success ↑), red for negative changes (collisions ↓ is good)
- **Summary dashboard**: `summary_dashboard.png` provides a comprehensive overview comparing all scenarios
- **Enhanced error bars**: Statistical confidence with mean ± standard deviation

### Notes

- `src/main.py` and `src/main_extended.py` are deprecated. Use `run_simulation.py` for all runs.
- Per-step CSV is useful for deep analysis; the aggregate CSV contains one row per run.

### Logs

- By default, logs are written to `logs/` at the project root (configurable via `output.logs_dir` or `--logs-dir`).
- Example: `python run_simulation.py --scenario highway --composition mixed --logs-dir tmp_logs`


