## Running Simulations via run_simulation.py

This guide explains how to launch simulations using the unified CLI entry point.

### Arguments

- --config: Optional. Path to unified `config.yaml` (defaults to project `config.yaml`).
- --scenario: Optional. One of: highway, merge, intersection, roundabout, racetrack. If omitted, uses enabled scenarios from `config.scenarios`.
- --composition: Optional. One of: selfish, cooperative, mixed. If omitted, uses compositions listed under the scenario in `config.scenarios`.
- --render: Optional flag. If set, enables on-screen visualization (render_mode=human).
- --output-csv: Optional path. Appends one row of run-level metrics to this CSV.
- --output-steps-csv: Optional path. Saves per-step metrics for a single run.
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

### Notes

- `src/main.py` and `src/main_extended.py` are deprecated. Use `run_simulation.py` for all runs.
- Per-step CSV is useful for deep analysis; the aggregate CSV contains one row per run.

### Logs

- By default, logs are written to `logs/` at the project root (configurable via `output.logs_dir` or `--logs-dir`).
- Example: `python run_simulation.py --scenario highway --composition mixed --logs-dir tmp_logs`


