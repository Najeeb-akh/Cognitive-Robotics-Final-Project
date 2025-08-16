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

from scenarios import get_scenario_configurations  # base scenarios
from scenarios_extended import (
    get_extended_scenario_configurations,
)
from metrics import MetricsCollector, MetricsAggregator
from main import (
    run_single_simulation as run_single_simulation_base,
    create_agent_policy as create_agent_policy_base,
)
from main_extended import (
    run_single_simulation as run_single_simulation_ext,
    create_agent_policy as create_agent_policy_ext,
    detect_scenario_type as detect_scenario_type_ext,
)


def setup_logging(log_level: str = 'INFO') -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(CURRENT_DIR, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        ],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified simulation runner')
    parser.add_argument('--scenario', required=True, help='Scenario to run (e.g., highway, merge, intersection, roundabout, racetrack)')
    parser.add_argument('--composition', required=True, help='Agent composition: selfish | cooperative | mixed')
    parser.add_argument('--render', action='store_true', help='Enable visualization (render_mode=human)')
    parser.add_argument('--output-csv', help='Path to aggregate results CSV (appends one row per run)')
    parser.add_argument('--output-steps-csv', help='Path to per-step CSV for a single run')
    parser.add_argument('--config-overrides', help='JSON string for configuration overrides')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def resolve_scenarios(args: argparse.Namespace):
    scenario_key = args.scenario.lower()

    # Use extended scenarios list which includes originals
    scenarios = get_extended_scenario_configurations()

    # Normalize composition selection
    comp = args.composition.lower()
    if comp not in ('selfish', 'cooperative', 'mixed'):
        raise ValueError('composition must be one of: selfish, cooperative, mixed')

    def comp_match(c: dict) -> bool:
        selfish = c.get('selfish_ratio', 0)
        cooperative = c.get('cooperative_ratio', 0)
        if comp == 'selfish':
            return selfish == 1.0 and cooperative == 0.0
        if comp == 'cooperative':
            return selfish == 0.0 and cooperative == 1.0
        return selfish == 0.5 and cooperative == 0.5

    # Filter by scenario name and composition
    filtered = [entry for entry in scenarios if scenario_key in entry[0].lower() and comp_match(entry[2])]
    if not filtered:
        raise ValueError('No matching scenario/composition found for provided arguments')
    return filtered


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Load and override config
    # Always start from base config file per plan
    base_config_path = os.path.join(CURRENT_DIR, 'config.yaml')
    config = load_config(base_config_path)
    config = apply_overrides(config, args.config_overrides)

    # Rendering
    render_args = {'enabled': args.render, 'mode': 'human' if args.render else 'rgb_array'}

    # Resolve runner and policy creators by scenario type

    # Prepare aggregator
    aggregator = MetricsAggregator()

    # Iterate over matching entries (should typically be one)
    for scenario_name, scenario_func, composition in resolve_scenarios(args):
        env = scenario_func(config, render_mode=render_args['mode'])
        metrics_collector = MetricsCollector(config)

        # Create appropriate policy for scenario
        scenario_type = detect_scenario_type_ext(scenario_name)
        if scenario_type in ('intersection', 'roundabout', 'racetrack'):
            agent_policy = create_agent_policy_ext(composition, config, scenario_type)
            runner = run_single_simulation_ext
        else:
            agent_policy = create_agent_policy_base(composition, config)
            runner = run_single_simulation_base

        # Run single simulation; always pass render flag per plan when requested
        run_metrics = runner(env, agent_policy, config, metrics_collector, render=args.render)

        # Optionally save per-step data
        if args.output_steps_csv:
            steps_df = metrics_collector.get_step_data_dataframe()
            os.makedirs(os.path.dirname(args.output_steps_csv), exist_ok=True)
            steps_df.to_csv(args.output_steps_csv, index=False)

        # Aggregate and optionally persist run-level metrics
        composition_desc = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
        aggregator.add_run_metrics(scenario_name.split('_')[0], composition_desc, run_metrics)

        # Append a single run row to output CSV if provided
        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            import pandas as pd
            row = {
                'scenario': scenario_name.split('_')[0],
                'composition': composition_desc,
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


