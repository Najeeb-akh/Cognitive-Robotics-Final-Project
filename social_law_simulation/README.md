# Decentralized Simulation of Social Laws in Highway Traffic

A Python implementation that simulates and compares the effects of cooperative driving behaviors in highway traffic using the highway-env library. This project implements a decentralized approach where agents make decisions based solely on local observations, allowing emergent cooperative behaviors to improve traffic flow and safety.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Social Laws Implementation](#social-laws-implementation)
- [Results and Analysis](#results-and-analysis)
- [Requirements](#requirements)

## Overview

This project compares two types of driving agents in highway traffic scenarios:

1. **Selfish Agents**: Follow standard IDM (Intelligent Driver Model) and MOBIL lane-changing models, focusing purely on personal velocity maximization
2. **Cooperative Agents**: Extend selfish behavior with three social laws designed to improve overall traffic flow

The simulation maintains strict **decentralization** - each agent makes decisions based only on its local highway-env observations, ensuring realistic emergent behaviors.

## Features

### Implemented Social Laws (FR3)

1. **Cooperative Merging (FR3.1)**: Agents decelerate to create gaps for merging vehicles
2. **Polite Yielding (FR3.2)**: Agents slow down to accommodate lane-changing vehicles  
3. **Phantom Jam Mitigation (FR3.3)**: Agents increase following distance in dense traffic to absorb speed variations

### Simulation Scenarios (FR4)

- **Highway Scenario**: Multi-lane straight highway testing general traffic flow
- **Merge Scenario**: Highway with merge ramp testing cooperative behaviors

### Agent Compositions

Each scenario runs with three different agent compositions:
- 100% Selfish Agents
- 100% Cooperative Agents  
- 50% Selfish / 50% Cooperative Mix

### Metrics Collection (FR5)

- **Efficiency**: Average speed, throughput
- **Safety**: Collision count, Time-to-Collision events
- **Stability**: Acceleration standard deviation
- **Cooperation**: Merge success rates

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project:
```bash
git clone <repository-url>
cd social_law_simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python src/main.py --help
```

## Configuration

The simulation is configured through `config.yaml`. Key parameters include:

### Simulation Parameters
```yaml
simulation:
  duration_steps: 1000      # Steps per simulation run
  num_runs_per_config: 5    # Runs to average over
```

### Social Law Parameters
```yaml
social_laws:
  cooperative_merging:
    enabled: true
    deceleration_factor: 0.8
    detection_distance: 30.0
  
  polite_yielding:
    enabled: true
    speed_reduction_factor: 0.9
    gap_creation_time: 2.0
    
  phantom_jam_mitigation:
    enabled: true
    density_threshold: 40
    increased_time_headway: 2.0
```

### Environment Settings
```yaml
environment:
  highway:
    lanes_count: 4
    vehicles_count: 50
  merge:
    lanes_count: 3
    vehicles_count: 40
```

## Usage

### Basic Execution

Run all scenarios and compositions:
```bash
python src/main.py
```

### Advanced Options

Run specific scenario:
```bash
python src/main.py --scenario Highway
python src/main.py --scenario Merge
```

Run specific agent composition:
```bash
python src/main.py --composition "100% Selfish"
python src/main.py --composition "100% Cooperative"
```

**Enable real-time visualization** (NEW!):
```bash
python src/main.py --render
python src/main.py --render --scenario Highway --composition "100% Cooperative"
```

Use custom configuration:
```bash
python src/main.py --config custom_config.yaml
```

Enable debug logging:
```bash
python src/main.py --log-level DEBUG
```

### Real-Time Visualization

The `--render` flag enables real-time visualization of the simulation:

- **Highway Scenario**: Watch vehicles navigate multi-lane traffic
- **Merge Scenario**: Observe merging behaviors and lane changes
- **Agent Interactions**: See how selfish vs cooperative agents behave differently
- **Visual Elements**:
  - Blue vehicles: Standard traffic agents
  - Red vehicle: Ego vehicle (main controlled agent)
  - Green areas: Target lanes and merge zones
  
**Controls**:
- Close window or press ESC/Q to stop simulation early
- Simulation runs at ~20 FPS for clear observation

**Examples**:
```bash
# Watch cooperative merging in action
python src/main.py --render --scenario Merge --composition "100% Cooperative"

# Compare selfish vs cooperative highway behavior
python src/main.py --render --scenario Highway --composition "50% Selfish"
```

**Easy Demo Script**:
For convenience, use the interactive demo script:
```bash
python demo_visualization.py
```
This provides a menu-driven interface to run different visualization scenarios without remembering command-line arguments.

### Output

The simulation generates:

1. **Results CSV** (`results/results.csv`): Aggregated metrics across all runs
2. **Comparison Plots** (`plots/`): Bar charts comparing metrics by agent composition
3. **Log Files**: Detailed execution logs with timestamps

## Project Structure

```
social_law_simulation/
├── src/
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── selfish_policy.py      # Baseline IDM+MOBIL implementation
│   │   └── cooperative_policy.py  # Social laws implementation
│   ├── scenarios.py               # Highway and merge scenario setup
│   ├── metrics.py                 # Metrics collection and analysis
│   ├── visualization.py           # Plot generation
│   └── main.py                    # Main simulation runner
├── config.yaml                    # Configuration parameters
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Social Laws Implementation

### FR3.1: Cooperative Merging

**Trigger**: Vehicle detects another agent in merge lane attempting to enter ahead

**Action**: Moderately decelerate to create safe merging gap

```python
def _check_cooperative_merging(self):
    merging_vehicle = self._detect_merging_vehicle(merge_lane_index)
    if merging_vehicle and self._is_merge_space_insufficient(merging_vehicle):
        return 4  # SLOWER action
```

### FR3.2: Polite Yielding

**Trigger**: Adjacent vehicle signals intent to change lanes in front

**Action**: Slightly reduce speed to create gap for lane changer

```python
def _check_polite_yielding(self):
    if self._detect_lane_change_intent(adjacent_lane_index):
        self.yielding_timer = self.GAP_CREATION_TIME
        return 4  # SLOWER action
```

### FR3.3: Phantom Jam Mitigation

**Trigger**: Local traffic density exceeds threshold (40 vehicles/km/lane)

**Action**: Increase IDM time headway parameter from 1.5s to 2.0s

```python
def _check_phantom_jam_mitigation(self):
    density = self._calculate_local_density()
    if density > self.DENSITY_THRESHOLD:
        self.TIME_HEADWAY = self.INCREASED_TIME_HEADWAY
```

## Results and Analysis

### Expected Outcomes

Based on traffic flow theory, cooperative agents should demonstrate:

- **Higher Average Speed**: Reduced congestion through smoother merging
- **Fewer Collisions**: Better gap management and collision avoidance
- **Lower Acceleration Variance**: More stable, less aggressive driving
- **Higher Merge Success**: Improved cooperation at merge points

### Interpreting Results

The `results.csv` file contains mean and standard deviation for all metrics:

- `avg_speed_mean`: Average vehicle speed across simulation
- `total_collisions_mean`: Average collision count per run
- `acceleration_std_mean`: Driving stability metric (lower is smoother)
- `merge_success_rate_mean`: Percentage of successful merges

Generated plots visualize these comparisons across agent compositions.

## Requirements

### Core Dependencies

- `highway-env>=1.8.0`: Traffic simulation environment
- `gymnasium>=0.28.0`: Reinforcement learning interface
- `numpy>=1.21.0`: Numerical computations
- `pandas>=1.3.0`: Data analysis and manipulation
- `matplotlib>=3.5.0`: Plotting and visualization
- `seaborn>=0.11.0`: Statistical visualization
- `pyyaml>=6.0`: Configuration file parsing

### System Requirements

- **Memory**: 4GB RAM minimum, 8GB recommended
- **CPU**: Multi-core processor recommended for parallel runs
- **Storage**: 1GB free space for results and logs
- **OS**: Windows 10+, macOS 10.14+, or Linux

### Performance Notes

- Single simulation run (1000 steps, ~50 agents): < 5 minutes
- Complete study (all scenarios/compositions): 30-60 minutes
- Memory usage: ~500MB during execution

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all dependencies installed via `pip install -r requirements.txt`

**Configuration Errors**: Validate YAML syntax in `config.yaml`

**Memory Issues**: Reduce `vehicles_count` or `duration_steps` in config

**Slow Performance**: Reduce `num_runs_per_config` for faster testing

### Debug Mode

Enable detailed logging:
```bash
python src/main.py --log-level DEBUG
```

This provides step-by-step execution details and metrics collection information.

---

## Citation

If you use this code in your research, please cite:

```
Decentralized Simulation of Social Laws in Highway Traffic
Implementation of cooperative driving behaviors using highway-env
```

## License

This project is released under the MIT License. See LICENSE file for details.