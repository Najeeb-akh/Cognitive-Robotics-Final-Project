import os
import sys
import json
import argparse
import logging
from datetime import datetime
import random

import yaml
import numpy as np

# Ensure src is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from metrics import MetricsCollector, MetricsAggregator
from main import (
    run_single_simulation as run_single_simulation_base,
    create_agent_policy as create_agent_policy_base,
)
from main_extended import (
    create_agent_policy as create_agent_policy_ext,
    detect_scenario_type as detect_scenario_type_ext,
)
from scenarios import (
    create_highway_scenario,
    create_merge_scenario,
)
from scenarios_extended import (
    create_intersection_scenario,
    create_roundabout_scenario,
    create_racetrack_scenario,
    get_extended_scenario_configurations,
)


def setup_logging(log_level: str = 'INFO', log_file_path: str | None = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path))
        except Exception as e:
            # Fallback to console-only logging
            print(f"Warning: failed to create log file at {log_file_path}: {e}")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(base_config: dict, overrides_json: str | None) -> dict:
    if not overrides_json:
        return base_config
    overrides = json.loads(overrides_json)

    def deep_update(d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return deep_update(dict(base_config), overrides)


# Map scenario keys to environment builder functions
BUILDER_MAP = {
    'highway': create_highway_scenario,
    'merge': create_merge_scenario,
    'intersection': create_intersection_scenario,
    'roundabout': create_roundabout_scenario,
    'racetrack': create_racetrack_scenario,
}


def normalize_scenario_key(s: str) -> str:
    return s.strip().lower().replace('-', '_')


def comp_name_to_ratio(comp_name: str) -> dict:
    name = comp_name.strip().lower()
    if name == 'selfish':
        return {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}
    if name == 'cooperative':
        return {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}
    # default to mixed
    return {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified simulation runner')
    parser.add_argument('--config', default=os.path.join(CURRENT_DIR, 'config.yaml'), help='Path to unified config file')
    parser.add_argument('--scenario', help='Scenario to run (e.g., highway, merge, intersection, roundabout, racetrack). If omitted, uses enabled scenarios from config.scenarios')
    parser.add_argument('--composition', help='Agent composition: selfish | cooperative | mixed | all. If omitted, uses scenario-defined compositions from config.scenarios')
    parser.add_argument('--runs', type=int, help='Number of runs per scenario-composition combination (overrides config)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed (default: 0)')
    parser.add_argument('--render', action='store_true', help='Enable visualization (render_mode=human)')
    parser.add_argument('--output-csv', help='Path to aggregate results CSV (appends one row per run)')
    parser.add_argument('--output-steps-csv', help='Path to per-step CSV for a single run')
    parser.add_argument('--output-results', help='Path to save aggregated results CSV (one file per invocation)')
    parser.add_argument('--save-plots', action='store_true', help='Generate and save comparison plots after simulation')
    # Debug/diagnostics options
    parser.add_argument('--log-actions', action='store_true', help='Log ego actions and speeds for early steps')
    parser.add_argument('--log-actions-steps', type=int, default=200, help='Number of initial steps to log when --log-actions is set')
    parser.add_argument('--log-ego-every', type=int, default=50, help='Log ego summary every N steps')
    parser.add_argument('--config-overrides', help='JSON string for configuration overrides')
    parser.add_argument('--logs-dir', help='Directory to write log files (overrides config.output.logs_dir)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def resolve_from_config(args: argparse.Namespace, config: dict):
    selections = []  # list[(scenario_key, composition_name)]

    # Determine scenarios
    if args.scenario:
        scenario_keys = [normalize_scenario_key(args.scenario)]
    else:
        registry = (config or {}).get('scenarios', {}) or {}
        if registry:
            scenario_keys = [normalize_scenario_key(k) for k, v in registry.items() if isinstance(v, dict) and v.get('enable', False)]
        else:
            # Fallback to legacy enumeration if no registry exists
            legacy = get_extended_scenario_configurations()
            scenario_keys = []
            for name, _, _ in legacy:
                for key in BUILDER_MAP.keys():
                    if key in name.lower() and key not in scenario_keys:
                        scenario_keys.append(key)

    # Determine compositions
    for sk in scenario_keys:
        if args.composition:
            if args.composition.strip().lower() == 'all':
                comp_names = ['selfish', 'cooperative', 'mixed']
            else:
                comp_names = [args.composition.strip().lower()]
        else:
            reg = (config or {}).get('scenarios', {}).get(sk, {}) if (config or {}).get('scenarios') else {}
            comp_names = [c.strip().lower() for c in reg.get('compositions', ['selfish', 'cooperative', 'mixed'])]
        for comp in comp_names:
            if comp not in ('selfish', 'cooperative', 'mixed'):
                logging.warning(f"Unknown composition '{comp}' for scenario '{sk}', skipping")
                continue
            selections.append((sk, comp))
    return selections


def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility across all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: Individual environment seeds are set per run

def apply_population_overrides_if_needed(env, composition_name: str, config: dict) -> None:
    """
    Apply population-level parameter overrides per composition, with fairness:
    - cooperative: apply cooperative overrides to all background vehicles
    - selfish: explicitly reset all background vehicles to baseline values
    - mixed: apply cooperative overrides to a fraction of background vehicles
    """
    try:
        name = (composition_name or '').strip().lower()

        unwrapped = getattr(env, 'unwrapped', env)
        road = getattr(unwrapped, 'road', None)
        ego = getattr(unwrapped, 'vehicle', None)
        if road is None or not hasattr(road, 'vehicles'):
            return

        # Baseline parameters
        baseline_root = (config or {}).get('baseline', {}) if isinstance(config, dict) else {}
        baseline_idm = baseline_root.get('idm', {}) if isinstance(baseline_root, dict) else {}
        baseline_mobil = baseline_root.get('mobil', {}) if isinstance(baseline_root, dict) else {}

        baseline_T = baseline_idm.get('time_headway', 1.5)
        baseline_politeness = baseline_mobil.get('politeness_factor', 0.1)

        # Cooperative override parameters
        overrides_root = (config or {}).get('cooperative_population_overrides', {})
        coop_over = overrides_root.get('baseline', {}) if isinstance(overrides_root, dict) else {}
        idm_over = (coop_over.get('idm', {}) if isinstance(coop_over, dict) else {})
        mobil_over = (coop_over.get('mobil', {}) if isinstance(coop_over, dict) else {})

        coop_T = idm_over.get('time_headway', None)
        coop_politeness = mobil_over.get('politeness_factor', None)

        # Prepare population lists
        background = [v for v in getattr(road, 'vehicles', []) if (ego is None or v is not ego)]
        total = len(background)
        if total == 0:
            return

        # Determine cooperative fraction
        if name == 'cooperative':
            coop_fraction = 1.0
        elif name == 'selfish':
            coop_fraction = 0.0
        else:  # 'mixed'
            ratios = comp_name_to_ratio(name)
            coop_fraction = float(ratios.get('cooperative_ratio', 0.5))

        # Choose vehicles to receive cooperative overrides (deterministic given seed)
        import random as _random
        idxs = list(range(total))
        _random.shuffle(idxs)
        coop_count = int(round(total * coop_fraction))
        coop_set = set(idxs[:coop_count])

        updated_coop = 0
        updated_base = 0

        def _set_T(vehicle, value):
            nonlocal updated_coop, updated_base
            if value is None:
                return
            if hasattr(vehicle, 'T'):
                try:
                    vehicle.T = float(value)
                    return True
                except Exception:
                    return False
            if hasattr(vehicle, 'time_headway'):
                try:
                    setattr(vehicle, 'time_headway', float(value))
                    return True
                except Exception:
                    return False
            return False

        def _set_politeness(vehicle, value):
            if value is None:
                return False
            applied = False
            if hasattr(vehicle, 'politeness'):
                try:
                    vehicle.politeness = float(value)
                    applied = True
                except Exception:
                    applied = False
            if not applied:
                lcm = getattr(vehicle, 'lane_change_model', None)
                if lcm is not None and hasattr(lcm, 'politeness'):
                    try:
                        lcm.politeness = float(value)
                        applied = True
                    except Exception:
                        applied = False
            return applied

        for i, v in enumerate(background):
            if i in coop_set:
                # Apply cooperative overrides
                t_ok = _set_T(v, coop_T)
                p_ok = _set_politeness(v, coop_politeness)
                if t_ok or p_ok:
                    updated_coop += 1
            else:
                # Ensure baseline
                t_ok = _set_T(v, baseline_T)
                p_ok = _set_politeness(v, baseline_politeness)
                if t_ok or p_ok:
                    updated_base += 1

        logging.info(
            f"Population overrides: cooperative_set={updated_coop}/{coop_count}, baseline_set={updated_base}/{total - coop_count} (composition={name})"
        )
    except Exception as e:
        logging.warning(f"Failed applying population overrides: {e}")


def main() -> None:
    args = parse_args()
    
    # Set global seeds for reproducibility
    set_seeds(args.seed)
    
    # Load and override config
    base_config_path = args.config
    config = load_config(base_config_path)
    config = apply_overrides(config, args.config_overrides)

    # Inject debug flags into config for downstream modules
    config.setdefault('debug', {})
    config['debug']['log_actions'] = bool(args.log_actions)
    config['debug']['log_actions_steps'] = int(args.log_actions_steps or 0)
    config['debug']['log_ego_every'] = int(args.log_ego_every or 0)

    # Resolve logs directory and initialize logging
    logs_dir = (
        args.logs_dir
        or (config.get('output', {}).get('logs_dir') if isinstance(config, dict) else None)
        or os.path.join(CURRENT_DIR, 'logs')
    )
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(CURRENT_DIR, logs_dir)
    log_file = os.path.join(logs_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(args.log_level, log_file)

    # Rendering
    render_args = {'enabled': args.render, 'mode': 'human' if args.render else 'rgb_array'}

    # Prepare aggregator
    aggregator = MetricsAggregator()

    # Resolve scenarios and compositions
    selections = resolve_from_config(args, config)
    if not selections:
        raise ValueError('No scenarios/compositions selected. Provide --scenario/--composition or enable scenarios in config.scenarios')

    # Determine number of runs
    runs = args.runs or config.get('simulation', {}).get('num_runs_per_config', 1)
    
    for scenario_key, composition_name in selections:
        builder = BUILDER_MAP.get(scenario_key)
        if builder is None:
            logging.warning(f"No builder found for scenario '{scenario_key}', skipping")
            continue

        logging.info(f"Running {runs} runs for {scenario_key} scenario with {composition_name} composition")
        
        # Policy selection based on scenario type (create once for reuse)
        scenario_type = detect_scenario_type_ext(scenario_key)
        composition = comp_name_to_ratio(composition_name)
        comp_desc = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
        
        # Run multiple instances
        for run_idx in range(runs):
            seed = args.seed + run_idx
            logging.debug(f"Run {run_idx + 1}/{runs} with seed {seed}")
            
            # Create environment for this run
            env = builder(config, render_mode=render_args['mode'])
            # Apply population-level cooperative overrides if needed (Option A)
            apply_population_overrides_if_needed(env, composition_name, config)

            # Reset with seed for reproducibility
            try:
                env.reset(seed=seed)
            except Exception as e:
                logging.warning(f"Failed to set seed {seed}: {e}")
                env.reset()

            # Propagate runtime context so metrics and logic know the composition and scenario
            try:
                config.setdefault('runtime', {})
                config['runtime']['composition'] = composition_name
                config['runtime']['scenario_type'] = scenario_type
            except Exception:
                pass

            metrics_collector = MetricsCollector(config)

            # Create policy for this run
            if scenario_type in ('intersection', 'roundabout', 'racetrack'):
                agent_policy = create_agent_policy_ext(composition, config, scenario_type)
            else:
                agent_policy = create_agent_policy_base(composition, config)


            # Always use base runner (supports stop/pause/progress)
            run_metrics = run_single_simulation_base(env, agent_policy, config, metrics_collector, render=args.render)
            
            # Add run index and seed to metrics
            run_metrics['run_index'] = run_idx
            run_metrics['seed'] = seed

            # Optionally save per-step data (only for single run scenarios)
            if args.output_steps_csv and runs == 1:
                steps_df = metrics_collector.get_step_data_dataframe()
                os.makedirs(os.path.dirname(args.output_steps_csv), exist_ok=True)
                steps_df.to_csv(args.output_steps_csv, index=False)

            # Aggregate run-level metrics
            aggregator.add_run_metrics(scenario_key.capitalize(), comp_desc, run_metrics)

            # Optionally append to CSV (per-run output)
            if args.output_csv:
                os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
                import pandas as pd
                row = {
                    'scenario': scenario_key.capitalize(),
                    'composition': comp_desc,
                    'run_index': run_idx,
                    'seed': seed,
                    **run_metrics,
                }
                if os.path.exists(args.output_csv):
                    existing = pd.read_csv(args.output_csv)
                    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    combined.to_csv(args.output_csv, index=False)
                else:
                    pd.DataFrame([row]).to_csv(args.output_csv, index=False)

            env.close()

    # Generate final outputs after all runs complete
    if args.output_results:
        logging.info("Saving aggregated results...")
        aggregated_df = aggregator.get_aggregated_results()
        if not aggregated_df.empty:
            os.makedirs(os.path.dirname(args.output_results), exist_ok=True)
            aggregated_df.to_csv(args.output_results, index=False)
            logging.info(f"Aggregated results saved to {args.output_results}")
        else:
            logging.warning("No data to aggregate for results output")
    
    if args.save_plots:
        logging.info("Generating comparison plots...")
        aggregated_df = aggregator.get_aggregated_results()
        if not aggregated_df.empty:
            # Import visualization here to avoid import issues
            from visualization import generate_comparison_plots
            
            # Use plots directory from config or default
            plots_dir = (config.get('output', {}).get('plots_dir', 'plots') 
                        if isinstance(config, dict) else 'plots')
            if not os.path.isabs(plots_dir):
                plots_dir = os.path.join(CURRENT_DIR, plots_dir)
            
            plot_paths = generate_comparison_plots(aggregated_df, plots_dir)
            logging.info(f"Generated {len(plot_paths)} plots in {plots_dir}")
            for path in plot_paths:
                logging.info(f"  - {os.path.basename(path)}")
        else:
            logging.warning("No data available for plotting")


if __name__ == '__main__':
    main()


