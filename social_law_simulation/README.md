# **Decentralized Social Laws Simulation for Highway Traffic**

## 1. Introduction

This project investigates the emergence of cooperative behavior in a fully decentralized multi-agent system simulating highway traffic. Using the `highway-env` library, we model and compare how "selfish" versus "cooperative" driving policies impact traffic dynamics across various scenarios. Agents operate solely on local observations, meaning there is no central controller. The goal is to analyze how simple, pre-defined social rules can lead to significant improvements in system-wide traffic efficiency, safety, and stability.

This README serves as the primary guide for understanding the project's architecture, running the simulations, and interpreting the results.

## 2. Table of Contents
- [1. Introduction](#1-introduction)
- [2. Table of Contents](#2-table-of-contents)
- [3. Core Features](#3-core-features)
- [4. How to Run the Project](#4-how-to-run-the-project)
  - [4.1. Installation](#41-installation)
  - [4.2. Running Simulations](#42-running-simulations)
  - [4.3. Expected Outputs](#43-expected-outputs)
- [5. Project Configuration (`config.yaml`)](#5-project-configuration-configyaml)
- [6. Scenarios and Policies](#6-scenarios-and-policies)
  - [6.1. Implemented Scenarios](#61-implemented-scenarios)
  - [6.2. Agent Policies](#62-agent-policies)
- [7. Metrics and Evaluation](#7-metrics-and-evaluation)
- [8. Project Structure](#8-project-structure)
- [9. Requirements](#9-requirements)
- [10. Troubleshooting](#10-troubleshooting)

## 3. Core Features

- **Single Entry Point**: All simulations are executed through `run_simulation.py`, which provides a command-line interface for full control.
- **Diverse Scenarios**: The simulation supports five distinct traffic scenarios: `highway`, `merge`, `intersection`, `roundabout`, and `racetrack`.
- **Configurable Agent Behaviors**: Easily switch between `selfish`, `cooperative`, and `mixed` agent populations to compare outcomes.
- **Comprehensive Metrics**: The project evaluates performance based on:
    - **Efficiency**: Average speed, throughput.
    - **Safety**: Number of collisions, time-to-collision (TTC).
    - **Stability**: Standard deviation of vehicle accelerations.
    - **Scenario-Specific Metrics**: Merge success rates, intersection wait times, etc.
- **Config-Driven Design**: A single `config.yaml` file controls all aspects of the simulation, from environment parameters to agent social laws and output paths.
- **Automated Results**: The runner automatically aggregates metrics into CSV files and generates comparison plots, saved to the `results/` and `plots/` directories.
- **Live Visualization**: Any simulation can be rendered in real-time by adding the `--render` flag.

## 4. How to Run the Project

Follow these steps to set up the environment and run the simulations.

### 4.1. Installation

First, clone the repository and install the required Python packages using the provided `requirements.txt` file.

```bash
# Clone the repository (if you haven't already)
# git clone ...

# Navigate to the project directory
cd social_law_simulation/

# Install dependencies
pip install -r requirements.txt
```

### 4.2. Running Simulations

The `run_simulation.py` script is the main entry point. Here are the most common use cases for evaluation.

#### **Option 1: Run All Pre-configured Simulations (Recommended)**

This is the simplest way to reproduce all results. It will run every enabled scenario and composition defined in `config.yaml` and generate the final plots and CSV data.

```bash
python run_simulation.py --config config.yaml
```

#### **Option 2: Run a Single Scenario with Live Rendering**

To visualize a specific scenario and agent composition, use the `--scenario` and `--composition` flags, and add `--render`. This is useful for observing agent behaviors directly.

```bash
# Example: Watch cooperative agents on the highway
python run_simulation.py --scenario highway --composition cooperative --render
```

#### **Option 3: Run a Specific Scenario and Save Detailed Data**

This command runs a single simulation and saves both the aggregate summary and a detailed per-step log, which is useful for granular analysis.

```bash
python run_simulation.py --scenario merge --composition mixed \
  --output-csv results/results.csv \
  --output-steps-csv results/merge_mixed_steps_data.csv
```

#### **Option 4: Temporarily Override a Configuration Parameter**

You can modify any parameter from `config.yaml` at runtime without editing the file by using the `--config-overrides` flag. This is useful for quick experiments.

```bash
# Example: Run the roundabout scenario for only 500 steps
python run_simulation.py --scenario roundabout --composition cooperative \
  --config-overrides '{"simulation": {"duration_steps": 500}}'
```

### 4.3. Expected Outputs

After running the simulations (especially Option 1), the following artifacts will be generated:

- **`results/results.csv`**: An aggregated CSV file containing one row of metrics for each simulation run.
- **`plots/`**: This directory will contain PNG plots comparing the performance of selfish, cooperative, and mixed compositions across all scenarios for key metrics like average speed, collisions, and stability.
- **`logs/`**: Contains detailed log files for debugging purposes.

## 5. Project Configuration (`config.yaml`)

The `config.yaml` file is the heart of the project, allowing you to control every aspect of the simulation without modifying the source code.

**Key Sections:**
- `simulation`: Global settings like the number of steps per simulation and the number of runs to average over.
- `scenarios`: Enables or disables entire scenarios and specifies which agent compositions to run for each.
- `environment`: Defines the physical properties of each scenario (e.g., number of lanes, vehicle counts).
- `compositions`: Defines the ratio of selfish to cooperative agents for different named compositions.
- `social_laws`: **This is where the core logic of cooperative behavior is defined.** You can enable/disable specific rules (e.g., polite yielding) and tune their parameters.
- `metrics`: Configures thresholds and parameters for calculating metrics (e.g., TTC threshold for safety).
- `output`: Specifies the directories for saving results, plots, and logs.

## 6. Scenarios and Policies

### 6.1. Implemented Scenarios

- **Highway**: A multi-lane highway for analyzing lane-changing behavior and phantom jam mitigation.
- **Merge**: A lane-merging scenario to test cooperation during high-congestion events.
- **Intersection**: A four-way intersection where agents must coordinate to avoid collisions and deadlocks.
- **Roundabout**: A circular intersection that requires yielding and smooth flow maintenance.
- **Racetrack**: A high-speed oval track for studying overtaking and slipstreaming behaviors.

### 6.2. Agent Policies

The behavior of each agent is governed by a policy. The key distinction is between selfish and cooperative behaviors.

- **`SelfishPolicy`**: This policy is based on standard models like IDM (Intelligent Driver Model) for longitudinal control and MOBIL (Minimizing Overall Braking Induced by Lane Changes) for lateral control. Agents act purely to maximize their own utility (e.g., speed) without regard for others.
- **`CooperativePolicy`**: This policy extends the selfish base with a set of "social laws" that encourage pro-social behavior. These include:
    - **Cooperative Merging**: Actively slowing down to create gaps for merging vehicles.
    - **Polite Yielding**: Reducing speed to facilitate smoother lane changes for others.
    - **Phantom Jam Mitigation**: Increasing headway in dense traffic to absorb speed variations and prevent jams.
    - **Scenario-Specific Laws**: Policies for turn-taking at intersections, yielding at roundabouts, and safe overtaking on the racetrack.

The specific implementations can be found in the `src/policies/` directory.

## 7. Metrics and Evaluation

The success of different policies is measured quantitatively.

- **Primary Metrics**:
  - `average_speed`: Measures traffic flow efficiency.
  - `collisions`: The primary safety metric, counting the total number of collisions.
  - `acceleration_stability`: The standard deviation of accelerations, where lower values indicate smoother, more stable traffic flow.
- **Scenario-Specific Metrics**:
  - `merge_success_rate`: The percentage of vehicles that successfully merge.
  - `intersection_waiting_time`: The average time vehicles spend waiting to cross.
- **Output Analysis**: The `src/visualization.py` script is used internally by the runner to generate plots from the final `results.csv` file, providing a clear visual comparison between the different agent compositions.

## 8. Project Structure

The project is organized to separate configuration, core logic, and outputs.

```
social_law_simulation/
├── config.yaml               # Main configuration file
├── run_simulation.py         # SINGLE ENTRY POINT for all simulations
├── requirements.txt          # Project dependencies
├── results/                  # Output directory for CSV data
├── plots/                    # Output directory for generated graphs
├── logs/                     # Output directory for log files
├── src/                      # Source code directory
│   ├── scenarios.py          # Environment builders for core scenarios
│   ├── scenarios_extended.py # ...for extended scenarios
│   ├── metrics.py            # Core metric calculation logic
│   ├── metrics_extended.py   # ...for scenario-specific metrics
│   ├── visualization.py      # Plotting and visualization utilities
│   └── policies/             # Directory containing all agent behavior logic
│       ├── selfish_policy.py
│       └── cooperative_policy.py
│       └── ... (scenario-specific policies)
└── tests/                    # Unit tests
```

## 9. Requirements

- Python 3.8+
- Key packages (see `requirements.txt` for a full list):
  - `highway-env`: The core simulation environment.
  - `numpy`, `pandas`: For data manipulation and analysis.
  - `matplotlib`, `seaborn`: For plotting.
  - `pyyaml`: For parsing the `config.yaml` file.
  - `pytest`: For running tests.

## 10. Troubleshooting

- **Import Errors**: If you encounter import errors, ensure you have installed all packages correctly by running `pip install -r requirements.txt`.
- **Rendering Issues on macOS**: Visualization can sometimes be problematic. If the rendering window fails to open, try running without the `--render` flag. The simulation will still run and produce data.
- **Slow Performance**: The simulation can be computationally intensive. To speed it up for quick tests, you can reduce `simulation.duration_steps` or `environment.*.vehicles_count` in `config.yaml`.