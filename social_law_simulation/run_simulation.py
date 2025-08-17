import os
import sys
import json
import argparse
import logging
from datetime import datetime

import yaml

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
    parser.add_argument('--composition', help='Agent composition: selfish | cooperative | mixed. If omitted, uses scenario-defined compositions from config.scenarios')
    parser.add_argument('--render', action='store_true', help='Enable visualization (render_mode=human)')
    parser.add_argument('--output-csv', help='Path to aggregate results CSV (appends one row per run)')
    parser.add_argument('--output-steps-csv', help='Path to per-step CSV for a single run')
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


def main() -> None:
    args = parse_args()
    # Load and override config
    base_config_path = args.config
    config = load_config(base_config_path)
    config = apply_overrides(config, args.config_overrides)

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

    for scenario_key, composition_name in selections:
        builder = BUILDER_MAP.get(scenario_key)
        if builder is None:
            logging.warning(f"No builder found for scenario '{scenario_key}', skipping")
            continue

        # Create environment
        env = builder(config, render_mode=render_args['mode'])
        metrics_collector = MetricsCollector(config)

        # Policy selection based on scenario type
        scenario_type = detect_scenario_type_ext(scenario_key)
        composition = comp_name_to_ratio(composition_name)
        if scenario_type in ('intersection', 'roundabout', 'racetrack'):
            agent_policy = create_agent_policy_ext(composition, config, scenario_type)
        else:
            agent_policy = create_agent_policy_base(composition, config)

        # Always use base runner (supports stop/pause/progress)
        run_metrics = run_single_simulation_base(env, agent_policy, config, metrics_collector, render=args.render)

        # Optionally save per-step data (single-run convenience)
        if args.output_steps_csv:
            steps_df = metrics_collector.get_step_data_dataframe()
            os.makedirs(os.path.dirname(args.output_steps_csv), exist_ok=True)
            steps_df.to_csv(args.output_steps_csv, index=False)

        # Aggregate and optionally persist run-level metrics
        comp_desc = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
        aggregator.add_run_metrics(scenario_key.capitalize(), comp_desc, run_metrics)

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            import pandas as pd
            row = {
                'scenario': scenario_key.capitalize(),
                'composition': comp_desc,
                **run_metrics,
            }
            if os.path.exists(args.output_csv):
                existing = pd.read_csv(args.output_csv)
                combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                combined.to_csv(args.output_csv, index=False)
            else:
                pd.DataFrame([row]).to_csv(args.output_csv, index=False)

        env.close()


if __name__ == '__main__':
    main()


