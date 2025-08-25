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
from simulation_core import run_single_simulation as run_single_simulation_base, DiscreteActionAdapter, ContinuousActionAdapter
from policy_factory import create_agent_policy, detect_scenario_type
from scenarios import (
    create_highway_scenario,
    create_merge_scenario,
    create_intersection_scenario,
    create_roundabout_scenario,
    create_racetrack_scenario,
    get_scenario_configurations,
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
    # Control cooperative population overrides
    parser.add_argument('--no-pop-override', action='store_true', help='Disable cooperative population overrides for background vehicles')
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
            legacy = get_scenario_configurations()
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

        # Allow disabling via config flag
        try:
            pop_cfg = (config or {}).get('cooperative_population_overrides', {})
            enabled_flag = True if not isinstance(pop_cfg, dict) else bool(pop_cfg.get('enabled', True))
            if not enabled_flag:
                logging.info("Population overrides disabled via config")
                return
        except Exception:
            pass

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
                return False
            # Direct attributes common in highway-env
            for attr in ('T', 'time_headway', 'tau', 'headway'):
                if hasattr(vehicle, attr):
                    try:
                        setattr(vehicle, attr, float(value))
                        return True
                    except Exception:
                        pass
            # Known nested controller attributes
            nested_candidates = (
                'idm', 'controller', 'car_following_controller', 'acc_controller',
                'longitudinal_controller', 'driver', 'path_controller'
            )
            for parent in nested_candidates:
                obj = getattr(vehicle, parent, None)
                if obj is None:
                    continue
                for attr in ('T', 'time_headway', 'tau', 'headway'):
                    if hasattr(obj, attr):
                        try:
                            setattr(obj, attr, float(value))
                            return True
                        except Exception:
                            pass
            # Heuristic deep scan of simple attributes one level down
            try:
                for name in dir(vehicle):
                    if name.startswith('_'):
                        continue
                    sub = getattr(vehicle, name, None)
                    if sub is None or isinstance(sub, (int, float, str, bool)):
                        continue
                    for attr in ('T', 'time_headway', 'tau', 'headway'):
                        if hasattr(sub, attr):
                            try:
                                setattr(sub, attr, float(value))
                                return True
                            except Exception:
                                pass
            except Exception:
                pass
            return False

        def _set_politeness(vehicle, value):
            if value is None:
                return False
            # Direct attribute
            if hasattr(vehicle, 'politeness'):
                try:
                    vehicle.politeness = float(value)
                    return True
                except Exception:
                    pass
            # Common nested lane change models
            nested_candidates = (
                'lane_change_model', 'lane_changing_controller', 'controller', 'driver'
            )
            for parent in nested_candidates:
                obj = getattr(vehicle, parent, None)
                if obj is None:
                    continue
                if hasattr(obj, 'politeness'):
                    try:
                        setattr(obj, 'politeness', float(value))
                        return True
                    except Exception:
                        pass
            # Heuristic deep scan
            try:
                for name in dir(vehicle):
                    if name.startswith('_'):
                        continue
                    sub = getattr(vehicle, name, None)
                    if sub is None or isinstance(sub, (int, float, str, bool)):
                        continue
                    if hasattr(sub, 'politeness'):
                        try:
                            setattr(sub, 'politeness', float(value))
                            return True
                        except Exception:
                            pass
            except Exception:
                pass
            return False

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


class DiscreteActionAdapter:
    """
    Adapter that translates policy's 5-action indexing
    [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
    to the environment's Discrete action space of size 3+.
    Missing actions (FASTER/SLOWER) degrade to IDLE.
    """
    def __init__(self, base_policy, env):
        self.base_policy = base_policy
        self._env = env
        self._build_mapping()

    def _build_mapping(self) -> None:
        try:
            cfg_obj = getattr(self._env, 'config', {})
            if not isinstance(cfg_obj, dict):
                cfg_obj = getattr(getattr(self._env, 'unwrapped', self._env), 'config', {}) or {}
            act_spec = cfg_obj.get('action', {}) if isinstance(cfg_obj, dict) else {}
            actions_list = None
            if isinstance(act_spec, dict):
                if (act_spec.get('type') or '').lower() == 'discretemetaaction':
                    actions_list = act_spec.get('actions', None)
            n = getattr(getattr(self._env, 'action_space', None), 'n', None)
            # highway-env default order is: 0=IDLE, 1=LANE_LEFT, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
            default_env_5 = ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]
            if not actions_list:
                if n == 3:
                    actions_list = ["IDLE", "LANE_LEFT", "LANE_RIGHT"]
                elif n and n >= 5:
                    actions_list = default_env_5[:n]
                else:
                    actions_list = default_env_5
            self._sem_to_idx = {a: i for i, a in enumerate(actions_list)}
            self._fallback = self._sem_to_idx.get("IDLE", 0)
            self._env_n = n or len(actions_list)
            # Policy emits in custom semantics order below;
            # always translate from this policy order to env order above
            self._policy_semantics = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
        except Exception:
            # Safe fallback to a minimal 3-action mapping with IDLE at index 0
            self._sem_to_idx = {"IDLE": 0, "LANE_LEFT": 1, "LANE_RIGHT": 2}
            self._fallback = 0
            self._env_n = 3
            self._policy_semantics = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

    def _translate(self, policy_action: int) -> int:
        try:
            # Always translate from policy semantics to environment index.
            # Policy order: [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER]
            if isinstance(policy_action, int) and 0 <= policy_action < len(self._policy_semantics):
                sem = self._policy_semantics[policy_action]
                return int(self._sem_to_idx.get(sem, self._fallback))
        except Exception:
            pass
        return int(self._fallback)

    def act(self, obs):
        action = None
        try:
            action = self.base_policy.act(obs)
        except Exception:
            action = None
        if isinstance(action, (int, np.integer)):
            return self._translate(int(action))
        # Unsupported type -> fallback
        return self._fallback


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
        scenario_type = detect_scenario_type(scenario_key)
        composition = comp_name_to_ratio(composition_name)
        comp_desc = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
        
        # Run multiple instances
        for run_idx in range(runs):
            seed = args.seed + run_idx
            logging.debug(f"Run {run_idx + 1}/{runs} with seed {seed}")
            
            # Create environment for this run
            env = builder(config, render_mode=render_args['mode'])
            # Validate action and observation spaces once
            try:
                # Observation validation (robust across highway-env versions)
                cfg_obj = getattr(env, 'config', {})
                if not isinstance(cfg_obj, dict):
                    cfg_obj = getattr(getattr(env, 'unwrapped', env), 'config', {}) or {}
                obs_spec = cfg_obj.get('observation', None)
                obs_type = None
                feats = None
                if isinstance(obs_spec, str):
                    obs_type = obs_spec
                elif isinstance(obs_spec, dict):
                    obs_type = obs_spec.get('type', None)
                    feats = obs_spec.get('features', None)
                # Assume Kinematics defaults if type indicates so and features missing
                if (obs_type or '').lower() == 'kinematics' and feats is None:
                    feats = ["presence", "x", "y", "vx", "vy"]
                # Validate minimally: required features should be present if list is provided
                required = {"presence", "x", "y", "vx", "vy"}
                if isinstance(feats, (list, tuple)):
                    missing = required.difference(set(feats))
                    if missing:
                        logging.warning(f"Observation missing expected features {sorted(missing)}; proceeding anyway with {feats}")
                else:
                    # No features provided; warn but continue
                    logging.warning(f"Observation features not specified; proceeding with default assumptions (got: {feats})")
                # Soft checks for other fields if present
                if isinstance(obs_spec, dict):
                    if 'absolute' in obs_spec and bool(obs_spec.get('absolute')):
                        logging.warning("Observation 'absolute' is True; relative coordinates are recommended")
                    if 'order' in obs_spec and obs_spec.get('order') != 'sorted':
                        logging.warning(f"Observation 'order' is {obs_spec.get('order')}; 'sorted' recommended")
                    if bool(obs_spec.get('normalize', False)):
                        logging.warning("Observation 'normalize' is True; policies may expect raw values")
                    if 'see_ego' in obs_spec and not bool(obs_spec.get('see_ego')):
                        logging.warning("Observation excludes ego (see_ego=False); policies may assume ego row present")
                # Action validation
                action_space = getattr(env, 'action_space', None)
                from gymnasium.spaces import Discrete, Box
                assert action_space is not None, "Missing action_space"
                if isinstance(action_space, Discrete):
                    if action_space.n < 5:
                        raise AssertionError(f"Discrete action space too small for 5-action policies: {action_space.n}")
                elif isinstance(action_space, Box):
                    # Expect 2D continuous control (steering, acceleration)
                    shape = getattr(action_space, 'shape', None)
                    assert shape and len(shape) == 1 and shape[0] >= 2, f"Unsupported Box action shape: {shape}"
                else:
                    raise AssertionError("Unsupported action space type; expected Discrete or Box")
            except AssertionError as e:
                logging.error(f"Environment interface assertion failed: {e}")
                raise

            # Apply class-level behavior overrides BEFORE reset so background vehicles inherit them
            try:
                from highway_env.vehicle.behavior import IDMVehicle
                if composition_name == 'cooperative':
                    IDMVehicle.TIME_WANTED = float((config.get('cooperative_population_overrides', {})
                                                    .get('baseline', {})
                                                    .get('idm', {})
                                                    .get('time_headway', 2.0)))
                    IDMVehicle.POLITENESS = float((config.get('cooperative_population_overrides', {})
                                                   .get('baseline', {})
                                                   .get('mobil', {})
                                                   .get('politeness_factor', 0.6)))
                    logging.info(f"Applied class-level overrides for cooperative: TIME_WANTED={IDMVehicle.TIME_WANTED}, POLITENESS={IDMVehicle.POLITENESS}")
                elif composition_name == 'selfish':
                    IDMVehicle.TIME_WANTED = float((config.get('baseline', {})
                                                    .get('idm', {})
                                                    .get('time_headway', 1.5)))
                    IDMVehicle.POLITENESS = float((config.get('baseline', {})
                                                   .get('mobil', {})
                                                   .get('politeness_factor', 0.1)))
                    logging.info(f"Applied class-level overrides for selfish: TIME_WANTED={IDMVehicle.TIME_WANTED}, POLITENESS={IDMVehicle.POLITENESS}")
                else:
                    # mixed: choose intermediate values (weighted) or baseline
                    base_T = float(config.get('baseline', {}).get('idm', {}).get('time_headway', 1.5))
                    coop_T = float((config.get('cooperative_population_overrides', {})
                                     .get('baseline', {}).get('idm', {}).get('time_headway', 2.0)))
                    IDMVehicle.TIME_WANTED = (base_T + coop_T) / 2.0
                    base_P = float(config.get('baseline', {}).get('mobil', {}).get('politeness_factor', 0.1))
                    coop_P = float((config.get('cooperative_population_overrides', {})
                                     .get('baseline', {}).get('mobil', {}).get('politeness_factor', 0.6)))
                    IDMVehicle.POLITENESS = (base_P + coop_P) / 2.0
                    logging.info(f"Applied class-level overrides for mixed: TIME_WANTED={IDMVehicle.TIME_WANTED}, POLITENESS={IDMVehicle.POLITENESS}")
            except Exception as e:
                logging.warning(f"Failed to apply class-level IDM/MOBIL overrides: {e}")

            # Reset with seed for reproducibility (must occur before any vehicle overrides)
            try:
                env.reset(seed=seed)
            except Exception as e:
                logging.warning(f"Failed to set seed {seed}: {e}")
                env.reset()

            # Apply population-level cooperative overrides after reset so vehicles exist
            if not bool(getattr(args, 'no_pop_override', False)):
                apply_population_overrides_if_needed(env, composition_name, config)

            # Propagate runtime context so metrics and logic know the composition and scenario
            try:
                config.setdefault('runtime', {})
                config['runtime']['composition'] = composition_name
                config['runtime']['scenario_type'] = scenario_type
            except Exception:
                pass

            metrics_collector = MetricsCollector(config)

            # Create policy for this run
            agent_policy = create_agent_policy(composition, config, scenario_type)
            try:
                logging.info(f"Ego policy: {agent_policy.__class__.__name__} (selfish_ratio={composition.get('selfish_ratio')})")
            except Exception:
                pass


            # Always use base runner (supports stop/pause/progress)
            # Inject runtime-derived parameters into policy (lane width, dt)
            try:
                unwrapped = getattr(env, 'unwrapped', env)
                road = getattr(unwrapped, 'road', None)
                vehicle = getattr(unwrapped, 'vehicle', None)
                width = None
                if road is not None and vehicle is not None:
                    net = getattr(road, 'network', None)
                    lane_index = getattr(vehicle, 'lane_index', None)
                    if net is not None and lane_index is not None:
                        try:
                            lane = net.get_lane(lane_index)
                            width = getattr(lane, 'width', None)
                        except Exception:
                            width = None
                if width is not None:
                    try:
                        setattr(agent_policy, '_lane_width_override', float(width))
                    except Exception:
                        pass
                    # If policy wraps a base policy, propagate
                    try:
                        base_pol = getattr(agent_policy, 'base_policy', None)
                        if base_pol is not None:
                            setattr(base_pol, '_lane_width_override', float(width))
                    except Exception:
                        pass
                # Set dt from env policy_frequency for time-aware timers
                try:
                    pf = None
                    cfg_obj = getattr(env, 'config', None)
                    if isinstance(cfg_obj, dict):
                        pf = cfg_obj.get('policy_frequency', None)
                    if pf is None:
                        cfg_obj2 = getattr(unwrapped, 'config', None)
                        if isinstance(cfg_obj2, dict):
                            pf = cfg_obj2.get('policy_frequency', None)
                    if pf:
                        dt = 1.0 / float(pf)
                        setattr(agent_policy, '_dt', dt)
                        base_pol = getattr(agent_policy, 'base_policy', None)
                        if base_pol is not None:
                            setattr(base_pol, '_dt', dt)
                except Exception:
                    pass
            except Exception:
                pass

            # Always wrap policy with adapter to ensure correct mapping to env action space
            try:
                from gymnasium.spaces import Discrete as _D, Box as _B
                a_sp = getattr(env, 'action_space', None)
                if isinstance(a_sp, _D):
                    agent_policy = DiscreteActionAdapter(agent_policy, env)
                elif isinstance(a_sp, _B):
                    agent_policy = ContinuousActionAdapter(agent_policy, env)
            except Exception:
                pass

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


