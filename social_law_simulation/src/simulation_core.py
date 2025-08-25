import logging
import numpy as np


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
            self._policy_semantics = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
        except Exception:
            self._sem_to_idx = {"IDLE": 0, "LANE_LEFT": 1, "LANE_RIGHT": 2}
            self._fallback = 0
            self._env_n = 3
            self._policy_semantics = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

    def _translate(self, policy_action: int) -> int:
        try:
            # Always translate from policy semantics to environment index.
            # Policy emits indices in order [LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER].
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
        return self._fallback


class ContinuousActionAdapter:
    """
    Adapter that maps the policy's 5-action semantics to a continuous Box
    action space (steering, acceleration). Heuristic mapping:
    - LANE_LEFT  -> steer left, neutral accel
    - IDLE       -> neutral steer, neutral accel
    - LANE_RIGHT -> steer right, neutral accel
    - FASTER     -> neutral steer, positive accel
    - SLOWER     -> neutral steer, negative accel
    """
    def __init__(self, base_policy, env):
        self.base_policy = base_policy
        self._env = env
        self._build_ranges()

    def _build_ranges(self):
        try:
            sp = getattr(self._env, 'action_space', None)
            low = getattr(sp, 'low', None)
            high = getattr(sp, 'high', None)
            if low is not None and high is not None and len(low) >= 2 and len(high) >= 2:
                self.steer_min, self.acc_min = float(low[0]), float(low[1])
                self.steer_max, self.acc_max = float(high[0]), float(high[1])
            else:
                raise ValueError('invalid action bounds')
        except Exception:
            # Safe defaults
            self.steer_min, self.steer_max = -1.0, 1.0
            self.acc_min, self.acc_max = -1.0, 1.0
        # Precompute neutral values
        self.steer_zero = 0.0
        self.acc_zero = 0.0
        self.acc_pos = self.acc_max * 0.7
        self.acc_neg = self.acc_min * 0.7
        self.steer_left = self.steer_min * 0.7
        self.steer_right = self.steer_max * 0.7

    def _map_semantic(self, sem_idx: int):
        # Policy semantics order must match DiscreteActionAdapter._policy_semantics
        if sem_idx == 0:  # LANE_LEFT
            return [self.steer_left, self.acc_zero]
        if sem_idx == 1:  # IDLE
            return [self.steer_zero, self.acc_zero]
        if sem_idx == 2:  # LANE_RIGHT
            return [self.steer_right, self.acc_zero]
        if sem_idx == 3:  # FASTER
            return [self.steer_zero, self.acc_pos]
        if sem_idx == 4:  # SLOWER
            return [self.steer_zero, self.acc_neg]
        return [self.steer_zero, self.acc_zero]

    def act(self, obs):
        try:
            action = self.base_policy.act(obs)
            if isinstance(action, (int, np.integer)):
                return self._map_semantic(int(action))
        except Exception:
            pass
        return [self.steer_zero, self.acc_zero]

def run_single_simulation(env, agent_policy, config, metrics_collector, render=False,
                          stop_event=None, pause_event=None, progress_cb=None):
    """
    Canonical single-run loop used by tests and runner.
    """
    duration_steps = config.get('simulation', {}).get('duration_steps', 1000)

    obs, info = env.reset()
    metrics_collector.reset_metrics()

    agent_name = agent_policy.__class__.__name__ if agent_policy else "DefaultPolicy"
    logging.info(f"Starting simulation with {agent_name} for {duration_steps} steps")

    try:
        import numpy as _np
        log_actions = bool(config.get('debug', {}).get('log_actions', False))
        log_actions_steps = int(config.get('debug', {}).get('log_actions_steps', 0))
        log_ego_every = int(config.get('debug', {}).get('log_ego_every', 0))
        for step in range(duration_steps):
            if stop_event and stop_event.is_set():
                logging.info(f"Simulation stopped by user at step {step}")
                break

            if pause_event and pause_event.is_set():
                import time
                while pause_event.is_set() and not (stop_event and stop_event.is_set()):
                    time.sleep(0.05)
                if stop_event and stop_event.is_set():
                    logging.info(f"Simulation stopped while paused at step {step}")
                    break

            if agent_policy:
                action = agent_policy.act(obs)
                if log_actions and step < log_actions_steps:
                    ego = getattr(env.unwrapped, 'vehicle', None)
                    ego_speed = getattr(ego, 'speed', None) if ego is not None else None
                    logging.info(f"[ACTIONS] step={step} action={action} ego_speed={ego_speed}")
            else:
                action = 1

            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()
                import time
                fps = config.get('visualization', {}).get('fps', 20)
                time.sleep(1.0 / fps)

            metrics_collector.collect_step_metrics(env, step, info)

            if log_ego_every and step % log_ego_every == 0:
                ego = getattr(env.unwrapped, 'vehicle', None)
                if ego is not None:
                    logging.info(
                        f"[EGO] step={step} speed={getattr(ego, 'speed', None)} lane_index={getattr(ego, 'lane_index', None)} crashed={getattr(ego, 'crashed', False)}"
                    )

            if progress_cb and step % 50 == 0:
                partial_metrics = {
                    'avg_speed_so_far': float(_np.mean(metrics_collector.speed_history)) if hasattr(metrics_collector, 'speed_history') and metrics_collector.speed_history else 0.0,
                    'total_collisions_so_far': int(getattr(metrics_collector, 'collisions', 0))
                }
                progress_cb(step, duration_steps, partial_metrics)

            if terminated or truncated:
                logging.info(f"Simulation ended early at step {step} - terminated: {terminated}, truncated: {truncated}")
                if info:
                    logging.info(f"Info: {info}")
                break

            if step % 200 == 0:
                logging.info(f"Step {step}/{duration_steps} completed")

    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        raise

    final_metrics = metrics_collector.calculate_final_metrics()
    logging.info("Simulation completed successfully")
    return final_metrics


