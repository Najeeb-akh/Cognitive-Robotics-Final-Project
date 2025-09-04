## 1. Overview

`run_simulation.py` serves as the unified command-line interface (CLI) for orchestrating multi-agent traffic simulations. It is designed to facilitate reproducible research by enabling batch execution, statistical aggregation, automated plotting, and fine-grained control over agent behaviors for comparative analysis.

## 2. Core Concepts: Experimental Design

The simulation framework is built on a hierarchy of behavioral models. Understanding these layers—from the fundamental microscopic traffic models to high-level compositions—is critical for designing valid experiments.

### 2.1. Baseline Agent Models (IDM & MOBIL)

The `selfish` agent behavior, which serves as the experimental control, is governed by two well-established, rule-based microscopic traffic models: the Intelligent Driver Model (IDM) for longitudinal control and the MOBIL model for lateral control.

*   **IDM (Intelligent Driver Model)**: This is a car-following model that dictates a vehicle's acceleration and deceleration. Its core function is to compute the ego vehicle's acceleration based on a continuous function that considers several variables:
    *   Current velocity (`v`)
    *   Desired free-flow velocity (`v₀`)
    *   Spacing (gap) to the leading vehicle (`s`)
    *   Velocity difference with the leading vehicle (`Δv`)
    *   Policy parameters such as desired time headway (`T`), maximum acceleration (`a`), and comfortable braking deceleration (`b`).

    The model's objective is to smoothly adjust the vehicle's speed to reach its desired velocity while maintaining a safe following distance from a preceding vehicle, thereby avoiding collisions and replicating realistic human driving behavior in traffic streams.

*   **MOBIL (Minimizing Overall Braking Induced by Lane changes)**: This is a lane-changing decision model that works in conjunction with a car-following model like IDM. MOBIL determines whether a vehicle should change from its current lane to a target lane (left or right) by evaluating two fundamental criteria:
    1.  **Safety Criterion**: A lane change is only permissible if it does not necessitate that the new follower vehicle in the target lane brake at a deceleration exceeding a safe limit. This is a mandatory, hard constraint that prevents cut-offs that would lead to collisions.
    2.  **Incentive Criterion**: A lane change is executed only if it offers a net "advantage" or "utility." This is calculated by summing the potential acceleration gain for the ego vehicle against the deceleration imposed on the new follower vehicle, moderated by a "politeness factor." The change is made if the ego vehicle's gain outweighs the disadvantage for the other agent, exceeding a predefined threshold.

Together, IDM and MOBIL form a robust foundation for simulating non-cooperative, yet collision-averse, traffic flow. Social laws are implemented as heuristics that augment or override the decisions made by these baseline models.

### 2.2. Composition (`--composition`)

The `--composition` argument defines the global distribution of agent policies within a simulation. It operates at a high level of abstraction.

*   `selfish`: A homogeneous population where all agents employ the baseline behavioral models (IDM/MOBIL) without any augmented cooperative heuristics. This serves as the experimental control group.
*   `cooperative`: A homogeneous population where all agents are endowed with the full suite of social laws enabled in the `config.yaml`. This represents the maximum potential cooperative effect.
*   `mixed`: A heterogeneous population with a stochastic distribution of `selfish` and `cooperative` agents, designed to model more realistic traffic environments.
*   `all`: A meta-argument that executes separate simulation batches for `selfish`, `cooperative`, and `mixed` compositions sequentially.

### 2.3. Social Law (`--social-law`)

The `--social-law` argument is a high-precision tool for **ablation studies**. It allows for the isolation and evaluation of a single behavioral heuristic.

When this flag is specified, it **overrides the `--composition` argument**. The simulation is instantiated with a homogeneous population of `selfish` agents, and then *only* the specified social law is layered on top of their baseline policy.

This mechanism is crucial for quantifying the marginal impact of a specific cooperative behavior, attributing performance changes directly to that heuristic rather than to the confounding effects of a full cooperative suite.

### 2.4. Behavioral Model Summary

| Invocation Mode | Base Agent Policy | Augmented Behavior | Primary Use Case |
| :--- | :--- | :--- | :--- |
| `--composition selfish` | IDM/MOBIL | None | Establish a performance baseline (control group). |
| `--composition cooperative` | IDM/MOBIL | All enabled social laws from config. | Evaluate the maximum system-wide impact of full cooperation. |
| `--composition mixed` | IDM/MOBIL | Random assignment of all-or-none social laws. | Analyze system stability and performance in heterogeneous populations. |
| `--social-law <name>` | IDM/MOBIL | **Only** the specified social law `<name>`. | Isolate and quantify the causal effect of a single behavioral heuristic. |

## 3. Command-Line Interface (CLI) Reference

### 3.1. Experiment Configuration
*   `--config`: Optional. Path to the `config.yaml` file. Defaults to the project root.
*   `--scenario`: Optional. Specifies the environment (`highway`, `merge`, `intersection`, etc.). If omitted, runs all scenarios enabled in `config.scenarios`.
*   `--composition`: Optional. Defines agent population (`selfish`, `cooperative`, `mixed`, `all`). If omitted, uses compositions defined under the scenario in the config.
*   `--social-law`: Optional. Isolates a single social law, overriding `--composition`.
*   `--config-overrides`: Optional. A JSON string for dynamically overriding configuration values, useful for parameter sweeps. Example: `--config-overrides '{"environment": {"roundabout": {"vehicles_count": 20}}}'`.

### 3.2. Execution Control & Reproducibility
*   `--runs`: Optional integer. The number of iterations per scenario-composition tuple. Critical for statistical significance. Overrides the value in the config.
*   `--seed`: Optional integer. The base random seed for PRNGs (`random`, `numpy.random`, environment). The seed for run `i` is calculated as `seed + i`, ensuring both intra- and inter-invocation reproducibility. Defaults to `0`.

### 3.3. Data Output & Artifacts
*   `--output-csv`: Optional path. Appends one row of run-level aggregate metrics to the specified CSV file. Suitable for simple, sequential data logging.
*   `--output-steps-csv`: Optional path. Generates a detailed CSV containing per-step metrics for a single run. Useful for fine-grained temporal analysis but generates large files.
*   `--output-results`: **(Recommended for Analysis)** Optional path. Generates a single CSV file per invocation containing aggregated results (mean, std, min, max) across all runs, grouped by scenario and composition.
*   `--save-plots`: Optional flag. Generates and saves a suite of comparison plots from the aggregated results, including statistical annotations.

### 3.4. Runtime & Diagnostics
*   `--render`: Optional flag. Enables on-screen rendering (`render_mode=human`) for qualitative analysis and debugging.
*   `--logs-dir`: Optional. Specifies the directory for log file output. Defaults to `output.logs_dir` from the config ("logs").

## 4. Research Workflows & Examples

### 4.1. Statistical Benchmarking of Compositions
To quantitatively compare `selfish`, `cooperative`, and `mixed` behaviors in a highway scenario across 50 runs for statistical power.

```bash
python run_simulation.py --scenario highway --composition all \
  --runs 50 --seed 42 \
  --output-results results/highway_comparison.csv --save-plots
```

### 4.2. Ablation Study of a Specific Social Law
To isolate and measure the precise impact of `cooperative_merging` against the `selfish` baseline.

```bash
# Execute the baseline runs
python run_simulation.py --scenario merge --composition selfish \
  --runs 30 --seed 123 --output-results results/merge_analysis.csv

# Execute the experimental runs with the isolated social law
python run_simulation.py --scenario merge --social-law cooperative_merging \
  --runs 30 --seed 123 --output-results results/merge_analysis.csv --save-plots```
*Note: Appending results to the same file and then generating plots provides a direct comparison.*

### 4.3. Parameter Sensitivity Analysis
To evaluate the effect of vehicle density in a roundabout without modifying the `config.yaml`.

```bash
# Low density
python run_simulation.py --scenario roundabout --composition cooperative \
  --config-overrides '{"environment": {"roundabout": {"vehicles_count": 10}}}' \
  --runs 20 --output-results results/roundabout_density_sweep.csv

# High density
python run_simulation.py --scenario roundabout --composition cooperative \
  --config-overrides '{"environment": {"roundabout": {"vehicles_count": 30}}}' \
  --runs 20 --output-results results/roundabout_density_sweep.csv --save-plots
```

## 5. Interpreting Output Artifacts

### 5.1. Aggregated Results CSV (`--output-results`)
This is the primary data source for analysis. It provides a structured summary with columns for each metric's mean, standard deviation, min, and max across all runs, properly grouped by experimental parameters (scenario, composition).

### 5.2. Generated Plots (`--save-plots`)
The plotting functionality is designed for rapid analysis and includes:
*   **Statistical Error Bars**: Visualizes mean ± standard deviation, conveying the variance in results.
*   **Sample Size Annotation**: Each bar is annotated with `N=X` to explicitly state the number of runs backing the statistic.
*   **Delta Annotations**: Percentage changes are automatically calculated and displayed against the `selfish` baseline, providing immediate insight into the performance impact (e.g., "+15.2%"). Deltas are color-coded for intuitive assessment.

## 6. Social Law Catalogue

The following social laws can be invoked with `--social-law <name>` for isolated testing.

#### Basic Traffic Management
*   **`cooperative_merging`**: Detects merging attempts and modulates longitudinal velocity to increase gaps for merging vehicles.
*   **`polite_yielding`**: Responds to adjacent vehicles' lane change indicators by reducing speed to facilitate smoother lateral movements.
*   **`phantom_jam_mitigation`**: Increases time headway (THW) in high-density conditions to dampen the propagation of velocity perturbations.

#### Advanced Cooperative Behaviors
*   **`polite_gap_provision`**: Proactively adjusts speed to create gaps for surrounding vehicles, moving beyond reactive yielding.
*   **`cooperative_turn_taking`**: Implements a turn-taking protocol at intersections to ensure fair resource allocation and prevent starvation.
*   **`adaptive_right_of_way`**: Dynamically adjusts right-of-way priority based on state information, such as vehicle waiting times.

#### Scenario-Specific Laws
*   **`entry_facilitation`**: (Roundabouts/Merges) Creates viable gaps for vehicles entering the main traffic stream.
*   **`smooth_flow_maintenance`**: Harmonizes local vehicle speeds to minimize acceleration/deceleration variance and maintain flow stability.
*   **`exit_courtesy`**: (Roundabouts) Assists exiting vehicles through early signaling and cooperative lane changing.

#### High-Speed Scenario Laws (Racetracks)
*   **`safe_overtaking_protocol`**: Enforces minimum speed differentials and safe distances during overtaking maneuvers.
*   **`defensive_positioning`**: Heuristic for yielding to faster vehicles by adopting a defensive lane position.
*   **`slipstream_cooperation`**: Coordinates alternating leadership to leverage aerodynamic drafting benefits.