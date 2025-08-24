## Decentralized Social Laws Simulation — Findings and Recommendations

### Overview

- **Goal**: Simplify the simulation workflow and strengthen methodology using highway-env best practices to compare scenarios with/without decentralized social laws.
- **Scope reviewed**: `run_simulation.py`, `src/scenarios.py`, `src/scenarios_extended.py`, `src/policies/*`, `src/metrics.py`, `src/metrics_extended.py`, `src/visualization.py`, `config.yaml`, tests, and docs.

### 1) Simplify the Simulation Workflow

#### A. Remove redundancy and centralize environment building
- **Problem**: Multiple builders and deprecated entry points (`src/main.py`, `src/main_extended.py`, `fixed_scenarios.py`) cause confusion and partial duplication. `run_simulation.py` imports deprecated functions for policy selection.
- **Recommendation**: Keep `run_simulation.py` as the single entry point. Remove imports from deprecated modules and consolidate scenario creation into one small utility mapping `scenario_key -> env_id` and applying `config.environment[scenario]`.
- **Benefits**: Fewer moving parts, easier maintenance, clearer flow.

```python
# runner_utils.py
import gymnasium as gym

ENV_ID = {
    "highway": "highway-v0",
    "merge": "merge-v0",
    "intersection": "intersection-v0",
    "roundabout": "roundabout-v0",
    "racetrack": "racetrack-v0",
}

def build_env(scenario_key: str, config: dict, render_mode: str):
    env_id = (config.get("scenarios", {}).get(scenario_key, {}).get("env_id")
              or ENV_ID[scenario_key])
    env = gym.make(env_id, render_mode=render_mode)
    # Merge a reasonable default with scenario-specific config
    default = {"observation": {"type": "Kinematics", "normalize": True},
               "action": {"type": "DiscreteMetaAction"}}
    env_cfg = {**default, **(config.get("environment", {}).get(scenario_key, {}) or {})}
    env.unwrapped.configure(env_cfg)
    return env
```

#### B. Clarify composition semantics (ego-only vs population-wide)
- **Problem**: “Composition” currently affects only the ego policy; background vehicles remain default IDM, which can mislead comparisons.
- **Recommendation (choose one):**
  - **Option A (simple, consistent)**: Treat `composition` as the ego policy only. If you want “cooperative population,” expose separate environment overrides (e.g., higher IDM time headway, higher MOBIL politeness) via a config block.
    - Implemented: `cooperative_population_overrides.baseline.idm.time_headway`, `mobil.politeness_factor` are applied at runtime to background vehicles when `composition=cooperative`.
  - **Option B (advanced)**: Increase `controlled_vehicles` and assign policies per ratio with a utility like `assign_policies(env, composition, config)`. This requires more wiring and careful stepping.
- **Benefits**: Transparent and fair comparisons; reproducible semantics.

```yaml
# Option A: Cooperative population via environment overrides
cooperative_population_overrides:
  baseline:
    idm:
      time_headway: 2.0
    mobil:
      politeness_factor: 0.3
```

#### C. Multi-run execution and aggregation in the runner
- **Problem**: `num_runs_per_config` exists in `config.yaml` but is not honored in `run_simulation.py` loops; aggregation is manual.
- **Recommendation**: Add `--runs` and `--seed`, loop over runs, call `env.reset(seed=...)`, and aggregate across runs. Write a single aggregated CSV per invocation.
- **Benefits**: Robust statistics and reproducibility in a single command.

```python
# in run_simulation.py (sketch)
parser.add_argument('--runs', type=int)
parser.add_argument('--seed', type=int, default=0)

runs = args.runs or config.get('simulation', {}).get('num_runs_per_config', 5)
for run_idx in range(runs):
    seed = args.seed + run_idx
    env = builder(config, render_mode=render_args['mode'])
    env.reset(seed=seed)
    run_metrics = run_single_simulation_base(...)
    aggregator.add_run_metrics(scenario_key.capitalize(), comp_desc, run_metrics)
```

#### D. Use scenario-specific metrics automatically
- **Problem**: `ExtendedMetricsCollector` is implemented but unused in the main loop.
- **Recommendation**: Select metrics collector via a small factory.
- **Benefits**: Richer metrics for `intersection`, `roundabout`, `racetrack` with no code duplication.

```python
from src.metrics_extended import create_metrics_collector
metrics_collector = create_metrics_collector(config, scenario_type)
```

#### E. Improve CLI ergonomics and outputs
- **Problem**: Users manually stitch outputs and plots; common overrides require JSON strings.
- **Recommendation**:
  - Add `--save-plots` to generate plots at the end using `aggregator.get_aggregated_results()`.
  - Add `--output-results` for a single aggregated CSV path; keep `--output-csv` for append-only workflows.
  - Optional shorthands: `--duration-steps`, `--vehicles-count`, `--lanes-count` to override the current scenario’s environment fields.
- **Benefits**: Fewer steps, consistent artifacts, easier experiments.

```bash
python run_simulation.py --config config.yaml --runs 10 --seed 123 \
  --save-plots --output-results results/results.csv
```

#### F. Inline policy selection; stop importing deprecated modules
- **Problem**: `run_simulation.py` imports policy selection from deprecated `main_extended`.
- **Recommendation**: Centralize policy selection in the runner.
- **Benefits**: Self-contained entry point; fewer hidden dependencies.

```python
# in run_simulation.py
from src.policies.selfish_policy import SelfishPolicy
from src.policies.cooperative_policy import CooperativePolicy
from src.policies.intersection_policy import IntersectionCooperativePolicy, IntersectionSelfishPolicy
from src.policies.roundabout_policy import RoundaboutCooperativePolicy, RoundaboutSelfishPolicy
from src.policies.racetrack_policy import RacetrackCooperativePolicy, RacetrackSelfishPolicy

def create_agent_policy(composition: dict, config: dict, scenario_type: str):
    selfish = composition.get("selfish_ratio", 0.5) > 0.5
    if scenario_type == "intersection":
        return (IntersectionSelfishPolicy if selfish else IntersectionCooperativePolicy)(config)
    if scenario_type == "roundabout":
        return (RoundaboutSelfishPolicy if selfish else RoundaboutCooperativePolicy)(config)
    if scenario_type == "racetrack":
        return (RacetrackSelfishPolicy if selfish else RacetrackCooperativePolicy)(config)
    return (SelfishPolicy if selfish else CooperativePolicy)(config)
```

#### G. Scenario registry clarity in config
- **Problem**: `scenarios.*.builder` is unused.
- **Recommendation**: Replace with `env_id` (optional; defaults to the canonical id) or remove the field.
- **Benefits**: One source of truth for environment selection.

```yaml
scenarios:
  highway:
    env_id: "highway-fast-v0"  # optional override
    enable: true
    compositions: [selfish, cooperative, mixed]
```

#### H. Seeds, logging, and reproducibility
- **Problem**: Seeds and RNG aren’t consistently applied; logs are per-run but not per-experiment.
- **Recommendation**: Seed `env`, `numpy.random`, and `random` per run; log scenario, composition, seed, run index; optionally write a `metadata.json`.
- **Benefits**: Reproducible science and better traceability.

### 2) Improvements Based on highway-env Documentation

Reference: [highway-env docs](https://highway-env.farama.org)

#### A. Prefer env IDs, fast variants, and wrappers
- **Problem**: Not leveraging fast envs and wrappers like `RecordVideo`.
- **Recommendation**: Use `gym.make(id)` and `env.unwrapped.configure(...)`; adopt `highway-fast-v0` for batch runs; expose `--record-video` to capture episodes headless.
- **Benefits**: Faster experiments and better artifact capture.

```python
from gymnasium.wrappers import RecordVideo
if args.record_video:
    video_dir = os.path.join(config['output']['results_dir'], 'videos')
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(env, video_dir, episode_trigger=lambda ep: True)
```

**Plan**
- **Implementation steps**:
  1. Update env builder to honor `scenarios.<key>.env_id` with fallback to canonical ids; prefer `highway-fast-v0` for batch runs in `config.yaml`.
  2. In `run_simulation.py`, add CLI flags: `--record-video` (bool) and `--video-dir` (default to `<results_dir>/videos`).
  3. Wrap environments with `RecordVideo` when `--record-video` is set and `--render` is not used; ensure `episode_trigger=lambda ep: True` records every run.
  4. Ensure seeding by calling `env.reset(seed=seed)` after wrapping.
  5. Update docs (`usage_guide.md`, `README.md`) with examples and note performance trade-offs of fast env variants.
  6. Add a smoke test that verifies the video directory is created when `--record-video` is supplied (skip validating video contents).
- **Acceptance criteria**:
  - CLI toggles between standard and fast envs via config; when `--record-video` is set, videos are saved to the desired directory; runs are reproducible with provided seeds.
- **Risks & mitigations**:
  - Video writing issues on headless macOS: recommend running without `--render` and use `RecordVideo`; document any backend requirements.
  - Fast env ids unavailable: fallback to canonical ids with a logged warning.
- **Effort**: ~1–2 hours.

#### B. Observation configuration flexibility
- **Problem**: Hard-coded Kinematics observations.
- **Recommendation**: Allow `config.environment.<scenario>.observation` to specify alternatives (e.g., `OccupancyGrid`).
- **Benefits**: Better expressiveness for intersection/roundabout.

```yaml
environment:
  intersection:
    observation:
      type: "OccupancyGrid"
      grid_size: [25, 25]
      features: ["presence", "vx", "vy"]
    duration: 60
```

**Plan**
- **Implementation steps**:
  1. In `build_env(...)`, only inject the default Kinematics observation when `observation` is absent; otherwise merge the user-specified observation dict.
  2. Keep `action: { type: DiscreteMetaAction }` unless overridden by config.
  3. Add a concrete observation example to `config.yaml` for `intersection` and document in `usage_guide.md`.
  4. Provide a `--config-overrides` example to switch observation types without editing files.
  5. Add a smoke test that creates an env with a custom observation dict and runs a few steps.
- **Acceptance criteria**:
  - Custom observation blocks are honored; default used only when unspecified.
- **Risks & mitigations**:
  - Some observation types need extra fields; validate known keys and warn on missing ones.
- **Effort**: ~1 hour.

#### C. Metrics: wire extended metrics and add a few cheap, useful signals
- **Problem**: Extended metrics not used; some key signals absent.
- **Recommendation**: Use `create_metrics_collector`; consider tracking lane-change count (ego), headway proxy, and near-offroad events.
- **Benefits**: Richer, scenario-relevant evaluation.

**Plan**
- **Implementation steps**:
  1. Swap in `create_metrics_collector(config, scenario_type)` in `run_simulation.py` for collector creation.
  2. Extend `src/metrics.py` to compute additional signals:
     - `ego_lane_changes`: track previous `lane_index` for ego (when available) and increment on change.
     - `avg_headway_proxy`: each step, estimate front gap divided by max(ego_speed, epsilon); average over steps.
     - `near_offroad_events`: count steps where ego lateral position approaches road bounds when `offroad_terminal` is False.
  3. Add these fields to `step_data` and aggregate to run-level metrics in `calculate_final_metrics()`.
  4. Update `src/visualization.py` to optionally plot these metrics when present; no-op if missing.
  5. Unit tests: stub minimal vehicle attributes to validate counters increment as expected.
- **Acceptance criteria**:
  - New keys exist in outputs: `ego_lane_changes`, `avg_headway_proxy`, `near_offroad_events`.
  - Extended metrics automatically apply for complex scenarios; regressions avoided in base scenarios.
- **Risks & mitigations**:
  - Env vehicle APIs differ; guard with `hasattr` and sensible fallbacks.
- **Effort**: ~2–3 hours.

#### D. Experimental design for fair comparisons
- **Problem**: Potential configuration drift between compositions; seeds differ.
- **Recommendation**: Fix seeds across compositions per scenario; keep identical environment parameters; document any population-level changes.
- **Benefits**: Credible claims about social laws’ impact.

**Plan**
- **Implementation steps**:
  1. For each scenario, precompute a list of seeds of length `runs` from `--seed` (e.g., `seed + i`).
  2. Iterate seeds outermost, compositions innermost so each composition uses the same seed before advancing to the next seed.
  3. Log `{scenario, composition, seed, run_idx}` at run start; include `seed` in per-run CSV rows.
  4. Write a `metadata.json` alongside aggregated results with timestamp, git commit (if available), CLI args, overrides, selected scenarios/compositions, and seed lists.
  5. Document the policy in `README.md` and `usage_guide.md`.
  6. Test: assert identical seed sequences across compositions for a given scenario and invocation.
- **Acceptance criteria**:
  - Seeds shared across compositions within scenario; persisted in outputs and metadata.
- **Risks & mitigations**:
  - Composition-dependent env reconfiguration can break comparability; avoid such changes or record them explicitly.
- **Effort**: ~1–2 hours.

#### E. Results pipeline and visualization
- **Problem**: Append-style CSVs can be messy; plots lack deltas vs baseline.
- **Recommendation**: Per-invocation aggregated CSV; optionally annotate plots with N and % change vs selfish baseline; add a small “summary dashboard” across scenarios.
- **Benefits**: Clear story and cleaner artifacts.

**Plan**
- **Implementation steps**:
  1. Add `--output-results` (path) and `--save-plots` (bool). After runs, write aggregated results via `MetricsAggregator.get_aggregated_results()` to a fresh CSV.
  2. If `--save-plots`, call `generate_comparison_plots` using the aggregated dataframe.
  3. Enhance plots:
     - Annotate bars with sample size N (derive from group counts).
     - For each scenario, compute deltas vs the selfish baseline; include ±% annotations on cooperative/mixed bars.
     - Optional combined “dashboard” figure comparing key KPIs across scenarios with delta labels.
  4. Docs: update `README.md` with saved artifact locations and example images.
  5. Tests: small synthetic dataframe to exercise plotting functions and ensure files are written.
- **Acceptance criteria**:
  - Single aggregated CSV per invocation; plots include N and delta annotations where baseline exists.
- **Risks & mitigations**:
  - Missing baseline: skip delta gracefully; small N: show N to convey uncertainty.
- **Effort**: ~2–3 hours.

### 3) Example Config and CLI

```yaml
simulation:
  duration_steps: 1000
  num_runs_per_config: 10
  seed: 42

scenarios:
  highway:
    env_id: "highway-fast-v0"
    enable: true
    compositions: [selfish, cooperative, mixed]
  merge:
    enable: true
    compositions: [selfish, cooperative, mixed]

environment:
  highway:
    lanes_count: 4
    vehicles_count: 50
    duration: 200
  merge:
    lanes_count: 3
    vehicles_count: 40
    duration: 200

baseline:
  idm:
    time_headway: 1.5
  mobil:
    politeness_factor: 0.1

cooperative_population_overrides:
  baseline:
    idm:
      time_headway: 2.0
    mobil:
      politeness_factor: 0.3

output:
  results_dir: "results"
  plots_dir: "plots"
  data_filename: "results.csv"
  logs_dir: "logs"
```

```bash
# Full batch with plots and aggregated results
python run_simulation.py --config config.yaml --runs 10 --seed 123 \
  --save-plots --output-results results/results.csv

# Quick override for a targeted run
python run_simulation.py --scenario roundabout --composition cooperative \
  --runs 5 --seed 10 \
  --config-overrides '{"environment": {"roundabout": {"vehicles_count": 30}}}'
```

### 4) Optional Parallelism

- **Problem**: Runtime increases with runs and scenarios.
- **Recommendation**: Add `--jobs N` and use `concurrent.futures.ProcessPoolExecutor` to parallelize runs (no rendering). Ensure unique seeds (`base_seed + worker_index`).
- **Benefits**: 2–5× speedup on multi-core machines.

### 5) Housekeeping

- Archive or remove `fixed_scenarios.py` and stop importing deprecated modules in the runner.
- Add minimal unit tests for `intersection`, `roundabout`, and `racetrack` mirroring `tests/test_runner.py`.
- Add a “Results/Release” section in `README.md` tracking the exact config used for headline figures.

### 6) Quick Wins Checklist

- **Unify env builder** in a `build_env(...)` utility.
- **Inline policy selection** in `run_simulation.py`.
- **Honor multi-run + seed** with per-invocation aggregation.
- **Use `create_metrics_collector`** for scenario-specific metrics.
- **Add `--save-plots`/`--output-results`** to streamline artifacts.
- **Clarify composition semantics** and implement Option A or B consistently.


