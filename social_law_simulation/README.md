# A Framework for Simulating Decentralized Social Laws in Multi-Agent Traffic

## 1. Abstract

This project provides a simulation framework for investigating emergent cooperative behavior in decentralized, multi-agent traffic systems. Built upon the `highway-env` library, it enables the analysis of how pre-defined social rules, encoded as policies, affect system-wide traffic dynamics (efficiency, safety, stability) when agents operate solely on local observations. A key feature of this framework is the ability to move beyond monolithic "selfish" vs. "cooperative" comparisons by allowing researchers to isolate and evaluate the marginal impact of individual social laws (e.g., `cooperative_merging`, `phantom_jam_mitigation`), facilitating rigorous ablation studies and comparative analyses.

For a comprehensive guide to all command-line options and advanced workflows, please refer to `usage_guide.md`.

## 2. Table of Contents
- [1. Abstract](#1-abstract)
- [2. Table of Contents](#2-table-of-contents)
- [3. Core Features](#3-core-features)
- [4. Getting Started](#4-getting-started)
  - [4.1. Installation](#41-installation)
  - [4.2. Running Simulations: Common Workflows](#42-running-simulations-common-workflows)
- [5. Architecture and Core Concepts](#5-architecture-and-core-concepts)
  - [5.1. Configuration (`config.yaml`)](#51-configuration-configyaml)
  - [5.2. Scenarios](#52-scenarios)
  - [5.3. Agent Policies and Social Laws](#53-agent-policies-and-social-laws)
- [6. Research Capabilities and Advanced Usage](#6-research-capabilities-and-advanced-usage)
  - [6.1. Workflow: Ablation Study](#61-workflow-ablation-study)
  - [6.2. Workflow: Cross-Scenario Analysis](#62-workflow-cross-scenario-analysis)
  - [6.3. Advanced: Runtime Parameter Overrides](#63-advanced-runtime-parameter-overrides)
- [7. Evaluation Metrics and Outputs](#7-evaluation-metrics-and-outputs)
  - [7.1. Key Metrics](#71-key-metrics)
  - [7.2. Expected Outputs](#72-expected-outputs)
- [8. Project Structure](#8-project-structure)
- [9. Requirements and Troubleshooting](#9-requirements-and-troubleshooting)
  - [9.1. Requirements](#91-requirements)
  - [9.2. Troubleshooting](#92-troubleshooting)

## 3. Core Features

- **Unified Entry Point**: A single, powerful CLI (`run_simulation.py`) for all experimental execution.
- **Modular Scenarios**: Test agent behaviors across five distinct environments: `highway`, `merge`, `intersection`, `roundabout`, and `racetrack`.
- **Precise Behavioral Control**: Define agent populations as `selfish`, `cooperative`, or `mixed`, or use the `--social-law` flag to isolate a single behavior for ablation studies.
- **Config-Driven Design**: All experimental parameters—from simulation duration and vehicle counts to social law physics—are managed in a central `config.yaml` file for reproducibility.
- **Automated Analysis Pipeline**: Automatically aggregates raw data into statistical summaries (CSV) and generates publication-ready plots with delta annotations and sample sizes.
- **Live Visualization**: Any simulation can be rendered on-screen via the `--render` flag for qualitative analysis and debugging.

## 4. Getting Started

### 4.1. Installation

Clone the repository and install the required Python dependencies.

```bash
# Navigate to the project directory
cd social_law_simulation/

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4.2. Running Simulations: Common Workflows

All simulations are launched via `run_simulation.py`.

#### **Workflow 1: Full Batch Analysis (Recommended)**
Execute all enabled scenarios and compositions from `config.yaml` with multiple runs for statistical significance. This is the standard method for reproducing a full set of results.

```bash
# Run 10 iterations per configuration and save aggregated results and plots
python run_simulation.py --config config.yaml --runs 10 --seed 42 \
  --output-results results/full_batch_results.csv --save-plots
```

#### **Workflow 2: Visualize a Specific Scenario**
Render a single simulation on-screen to qualitatively observe agent interactions.

```bash
# Watch how cooperative agents navigate a busy merge
python run_simulation.py --scenario merge --composition cooperative --render
```

#### **Workflow 3: Isolate and Analyze a Single Social Law**
Use the `--social-law` flag to conduct a focused experiment on one specific behavior.

```bash
# Evaluate the impact of "cooperative_merging" across 20 runs
python run_simulation.py --scenario highway --social-law cooperative_merging --runs 20 \
  --output-results results/merging_analysis.csv --save-plots
```

## 5. Architecture and Core Concepts

### 5.1. Configuration (`config.yaml`)

This file is the central control panel for the entire framework. It is structured to manage:
- `simulation`: Global parameters like `duration_steps` and default `runs`.
- `scenarios`: Toggles for enabling scenarios and defining which compositions to run for each.
- `environment`: Physical parameters for each scenario (e.g., `vehicles_count`, `lanes_count`).
- `social_laws`: The **behavioral definition section**. Enable/disable specific laws and tune their underlying parameters.
- `output`: Default paths for logs, results, and plots.

### 5.2. Scenarios

The framework provides five distinct environments to test the robustness and generalizability of agent policies:
- **Highway**: Multi-lane environment for studying lane-changing and phantom jam mitigation.
- **Merge**: High-congestion merging scenario to test negotiation and gap creation.
- **Intersection**: Four-way intersection for analyzing collision avoidance and deadlock resolution.
- **Roundabout**: Circular intersection requiring yielding and smooth flow maintenance.
- **Racetrack**: High-speed oval for evaluating overtaking and strategic positioning.

### 5.3. Agent Policies and Social Laws

Agent behavior is determined by a policy. The system architecture is designed to facilitate clear comparisons.

#### Policy Implementations
- **`SelfishPolicy`**: The baseline agent using standard microscopic traffic models: the Intelligent Driver Model (IDM) for longitudinal control and MOBIL for lateral (lane-change) decisions. This policy serves as the experimental control group.
- **`CooperativePolicy`**: Extends the `SelfishPolicy` by enabling the **full suite** of active social laws defined in `config.yaml`.
- **`SingleSocialLawPolicy`**: A specialized policy that extends the `SelfishPolicy` with **only one** specified social law. This is the mechanism that powers the `--social-law` flag for isolated analysis.

#### Policy Selection via CLI

| Command | Effective Agent Policy | Use Case |
| :--- | :--- | :--- |
| `--composition selfish` | `SelfishPolicy` (pure IDM/MOBIL) | Establish a non-cooperative baseline. |
| `--composition cooperative` | `CooperativePolicy` (all active laws) | Evaluate the maximum impact of cooperation. |
| `--composition mixed` | Random mix of `Selfish` & `Cooperative` | Test robustness in heterogeneous populations. |
| `--social-law <name>` | `SingleSocialLawPolicy` (Selfish + one law) | Isolate and quantify the effect of one behavior. |

#### Available Social Laws (12 Total)
The framework includes a comprehensive library of social laws, categorized by function:
- **Basic Traffic Management**: `cooperative_merging`, `polite_yielding`, `phantom_jam_mitigation`.
- **Advanced Behaviors**: `polite_gap_provision`, `cooperative_turn_taking`, `adaptive_right_of_way`.
- **Scenario-Specific**: `entry_facilitation`, `smooth_flow_maintenance`, `exit_courtesy`.
- **High-Speed (Racetrack)**: `safe_overtaking_protocol`, `defensive_positioning`, `slipstream_cooperation`.

## 6. Research Capabilities and Advanced Usage

The CLI is designed to facilitate common research workflows.

### 6.1. Workflow: Ablation Study
To systematically quantify the marginal contribution of each social law relative to a baseline.

```bash
# 1. Establish the baseline performance (30 runs)
python run_simulation.py --scenario intersection --composition selfish --runs 30 \
  --output-results results/ablation.csv

# 2. Evaluate each law in isolation (appends to the same results file)
for law in cooperative_turn_taking adaptive_right_of_way; do
  python run_simulation.py --scenario intersection --social-law $law --runs 30 \
    --output-results results/ablation.csv
done

# 3. Generate comparative plots from the combined data
python run_simulation.py --scenario intersection --composition selfish --runs 0 \
  --output-results results/ablation.csv --save-plots
```

### 6.2. Workflow: Cross-Scenario Analysis
To assess the generalizability of a social law across different environments.

```bash
# Test 'polite_yielding' on both highway and merge scenarios
for scn in highway merge; do
  python run_simulation.py --scenario $scn --social-law polite_yielding --runs 20 \
    --output-results results/yielding_analysis_${scn}.csv --save-plots
done
```

### 6.3. Advanced: Runtime Parameter Overrides
Quickly test a hypothesis by modifying a configuration parameter without editing `config.yaml`.

```bash
# Test the effect of higher traffic density in the roundabout
python run_simulation.py --scenario roundabout --composition cooperative \
  --config-overrides '{"environment": {"roundabout": {"vehicles_count": 30}}}'
```

## 7. Evaluation Metrics and Outputs

### 7.1. Key Metrics
Performance is quantified using a standard set of metrics:
- **Efficiency**: `average_speed`, `throughput`.
- **Safety**: `collisions`, `time_to_collision` (TTC).
- **Stability**: `acceleration_stability` (std. dev. of accelerations).
- **Scenario-Specific**: `merge_success_rate`, `intersection_waiting_time`.

### 7.2. Expected Outputs
- **`results/`**: Contains CSV files with aggregated statistical data (mean, std, N) for each experimental group.
- **`plots/`**: Contains PNG plots visualizing the results. Key features include:
  - **Delta Annotations**: Percentage change relative to the `selfish` baseline.
  - **Sample Size (N=X)**: Number of runs per bar.
  - **Statistical Error Bars**: Indicating variance in results.
  - **Summary Dashboards**: Multi-panel plots comparing key metrics across all scenarios.
- **`logs/`**: Detailed log files for debugging.

## 8. Project Structure

```
social_law_simulation/
├── config.yaml               # Main configuration file
├── run_simulation.py         # SINGLE ENTRY POINT for all simulations
├── usage_guide.md            # Comprehensive CLI usage guide
├── requirements.txt          # Project dependencies
├── results/                  # Output directory for CSV data
├── plots/                    # Output directory for generated graphs
├── logs/                     # Output directory for log files
└── src/                      # Source code directory
    ├── scenarios.py          # Environment builders
    ├── simulation_core.py    # Core simulation engine
    ├── metrics.py            # Metric calculation logic
    ├── visualization.py      # Plotting utilities
    └── policies/             # Directory for all agent policy implementations
```

## 9. Requirements and Troubleshooting

### 9.1. Requirements
- Python 3.8+
- Key packages are listed in `requirements.txt`:
  - `highway-env`: The core simulation environment.
  - `numpy`, `pandas`: For data processing.
  - `matplotlib`, `seaborn`: For plotting.
  - `pyyaml`: For parsing the configuration.

### 9.2. Troubleshooting
- **Import Errors**: Ensure dependencies are correctly installed from `requirements.txt` inside your active virtual environment.
- **Rendering Issues**: If the visualization window fails (common on some OS/server setups), run without the `--render` flag. Data generation and analysis are independent of rendering.
- **Slow Performance**: For faster iterations, reduce `simulation.duration_steps` or `environment.*.vehicles_count` in `config.yaml` or use the `--config-overrides` flag.
- **Invalid Social Law**: If you receive an "Unknown social law" error, the message will list all valid names. Check your spelling or refer to the list in section 5.3.