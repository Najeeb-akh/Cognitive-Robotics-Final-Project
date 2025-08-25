"""
Metrics Collection System Implementation (FR5)

This module handles the collection and processing of all simulation metrics:
- Efficiency: Average speed, throughput
- Safety: Collisions, Time-to-Collision (TTC)
- Stability: Acceleration standard deviation  
- Cooperation: Merge success rate

All metrics are collected during simulation and aggregated for analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import math


class MetricsCollector:
    """
    Collects and processes simulation metrics according to FR5 requirements.
    """
    
    def __init__(self, config):
        """
        Initialize metrics collector with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Metrics configuration
        metrics_config = config.get('metrics', {})
        self.TTC_THRESHOLD = metrics_config.get('safety', {}).get('ttc_threshold', 2.0)
        self.COLLISION_DISTANCE = metrics_config.get('safety', {}).get('collision_distance', 1.5)
        self.TARGET_VELOCITY = metrics_config.get('efficiency', {}).get('target_velocity', 30.0)
        self.ACCEL_WINDOW = metrics_config.get('stability', {}).get('acceleration_window', 100)
        
        # Data storage
        self.reset_metrics()
        # Scenario/runtime context (may be injected by runner)
        runtime = config.get('runtime', {}) if isinstance(config, dict) else {}
        self.scenario_type = (runtime.get('scenario_type') or '').strip().lower()
        # Time step estimation (seconds per decision step)
        self._dt = None
        try:
            if isinstance(config, dict):
                sim_cfg = config.get('simulation', {}) or {}
                pf = sim_cfg.get('policy_frequency', None)
                if pf is not None:
                    pf = float(pf)
                    if pf > 0:
                        self._dt = 1.0 / pf
        except Exception:
            self._dt = None
    
    def reset_metrics(self):
        """Reset all metrics for a new simulation run."""
        # Efficiency metrics
        self.speed_history = []
        self.network_speed_history = []
        self.completed_routes = 0
        self.total_vehicles_spawned = 0
        self.vehicle_steps = 0
        
        # Safety metrics
        self.collisions = 0
        self.ttc_events = []
        self.min_ttc_per_step = []
        self.thw_values = []
        self.drac_values = []
        self.thw_violations = 0
        self.thw_violation_threshold = 1.0
        # Collision severity events (relative-speed squared proxy)
        self.collision_events = []
        # Ego-specific safety
        self.ego_collisions = 0
        self._ego_collision_counted = False
        
        # Stability metrics
        self.acceleration_history = defaultdict(list)
        self._last_speed_by_vid = {}
        self._last_accel_by_vid = {}
        self.jerk_values = []
        
        # Cooperation metrics (for merge scenarios)
        self.merge_attempts = 0
        self.successful_merges = 0
        
        # Per-step data for detailed analysis
        self.step_data = []
        # Lane-change quality
        self._prev_lane_index = {}
        self.lane_change_attempts = 0
        self.unsafe_lane_changes = 0
        # Right-lane compliance (ego-focused to start)
        self.right_lane_eligible_ticks = 0
        self.right_lane_compliant_ticks = 0
        # Throughput tracking (crossing lane ends)
        self._last_s_by_vid = {}
        self._lane_end_threshold = 5.0
    
    def collect_step_metrics(self, env, step_num, info=None):
        """
        Collect metrics for a single simulation step.
        
        Args:
            env: The highway-env environment
            step_num (int): Current simulation step
            info (dict|None): Optional info dictionary returned by env.step()
        """
        # Incorporate environment-provided info first for reliability
        self.collect_step_info(info)
        # Identify ego properly: prefer env.vehicle or controlled_vehicles[0]
        unwrapped = getattr(env, 'unwrapped', env)
        ego_vehicle = getattr(unwrapped, 'vehicle', None)
        if ego_vehicle is None and hasattr(unwrapped, 'controlled_vehicles') and unwrapped.controlled_vehicles:
            ego_vehicle = unwrapped.controlled_vehicles[0]

        road = unwrapped.road
        vehicles = road.vehicles if hasattr(road, 'vehicles') else []
        
        if not vehicles:
            return
        
        # Collect speed for efficiency metrics from the true ego vehicle only
        if ego_vehicle is not None and hasattr(ego_vehicle, 'speed') and not getattr(ego_vehicle, 'crashed', False):
            self.speed_history.append(ego_vehicle.speed)
        
        # Collect network-wide average speed (non-crashed vehicles)
        try:
            speeds = [float(getattr(v, 'speed', 0.0)) for v in vehicles if hasattr(v, 'speed') and not getattr(v, 'crashed', False)]
            if speeds:
                self.network_speed_history.append(float(np.mean(speeds)))
        except Exception:
            pass
        
        # Accumulate vehicle-steps for crash rate normalization
        try:
            self.vehicle_steps += int(len([v for v in vehicles if not getattr(v, 'crashed', False)]))
        except Exception:
            pass
        
        # Estimate acceleration and jerk using finite differences of speed per vehicle
        # Estimate dt if unknown from env config
        if self._dt is None:
            try:
                cfg = getattr(env, 'config', {})
                if not isinstance(cfg, dict):
                    cfg = getattr(getattr(env, 'unwrapped', env), 'config', {}) or {}
                # Prefer env policy_frequency; if missing, default to 1 Hz (Gymnasium policy default)
                pf = cfg.get('policy_frequency', None)
                if pf is None:
                    pf = 1.0
                pf = float(pf)
                if pf > 0:
                    self._dt = 1.0 / pf
            except Exception:
                self._dt = None
        for vehicle in vehicles:
            if getattr(vehicle, 'crashed', False):
                continue
            try:
                vid = getattr(vehicle, 'id', None) or id(vehicle)
                speed_now = float(getattr(vehicle, 'speed', 0.0))
                prev_speed = self._last_speed_by_vid.get(vid, None)
                accel_now = None
                if prev_speed is not None and self._dt:
                    accel_now = (speed_now - prev_speed) / self._dt
                    self.acceleration_history[vid].append(accel_now)
                    prev_accel = self._last_accel_by_vid.get(vid, None)
                    if prev_accel is not None:
                        jerk = (accel_now - prev_accel) / self._dt
                        # Clip extreme outliers to guard against dt mis-estimation
                        if np.isfinite(jerk):
                            self.jerk_values.append(float(jerk))
                    self._last_accel_by_vid[vid] = accel_now
                self._last_speed_by_vid[vid] = speed_now
            except Exception:
                continue
        
        # Check for collisions (transition-based) and compute severity
        self._check_collisions(vehicles, road, step_num)
        # Ego collision tracking
        if ego_vehicle is not None and getattr(ego_vehicle, 'crashed', False) and not self._ego_collision_counted:
            self.ego_collisions = 1
            self._ego_collision_counted = True

        # Periodic debug dump of ego state
        debug = self.config.get('debug', {}) if isinstance(self.config, dict) else {}
        log_every = int(debug.get('log_ego_every', 0) or 0)
        if log_every and step_num % log_every == 0:
            if ego_vehicle is not None:
                try:
                    lane_index = getattr(ego_vehicle, 'lane_index', None)
                except Exception:
                    lane_index = None
                print(f"[EGO-DBG] step={step_num} speed={getattr(ego_vehicle, 'speed', None)} lane_index={lane_index} crashed={getattr(ego_vehicle, 'crashed', False)}")
        
        # Calculate lane-aware Time-to-Collision and headway/DRAC using nearest neighbors
        self._calculate_neighbor_safety_events(road, vehicles)

        # Throughput: count vehicles crossing lane end zones (only for highway-like scenarios)
        if (self.scenario_type in (None, '', 'highway', 'merge')):
            self._update_throughput(road, vehicles)

        # Lane change quality and right-lane compliance (ego-focused)
        self._update_lane_change_quality(road, vehicles)
        self._update_right_lane_compliance(road, ego_vehicle)
        
        # Check merge events (for merge scenarios)
        self._check_merge_events(env, vehicles)
        
        # Store step data
        ego_speed = 0
        if ego_vehicle is not None and hasattr(ego_vehicle, 'speed') and not getattr(ego_vehicle, 'crashed', False):
            ego_speed = ego_vehicle.speed
                
        step_data = {
            'step': step_num,
            'num_vehicles': len(vehicles),
            'ego_speed': ego_speed,
            'collisions_this_step': 0  # Will be updated by collision detection
        }
        # Add selected info-derived fields when available
        if isinstance(info, dict):
            if 'speed' in info:
                step_data['info_speed'] = info.get('speed')
            if 'crashed' in info:
                step_data['info_crashed'] = bool(info.get('crashed'))
        self.step_data.append(step_data)

    def collect_step_info(self, info):
        """
        Parse the info dictionary from env.step() to enrich metrics.
        
        Args:
            info (dict|None): Info dict returned by env.step()
        """
        if not isinstance(info, dict):
            return
        # Optionally record collisions from info without double-counting
        crashed_flag = info.get('crashed')
        if isinstance(crashed_flag, (bool, np.bool_)):
            # Only update the per-step field; collision totals are handled elsewhere
            pass
        # Optionally record speed provided in info
        info_speed = info.get('speed')
        try:
            if info_speed is not None:
                float(info_speed)  # validate numeric
                # Prefer ego speed from env for aggregates; info speed stored per-step
                pass
        except (TypeError, ValueError):
            pass
    
    def _check_collisions(self, vehicles, road, step_num):
        """
        Check for collisions using highway-env's built-in crash detection.
        
        Args:
            vehicles (list): List of vehicles in simulation
            road: Road object for neighbor queries
            step_num (int): Current step number
        """
        collisions_this_step = 0
        # Maintain per-vehicle crash state to detect transitions
        if not hasattr(self, '_vehicle_crash_state'):
            self._vehicle_crash_state = {}
        if not hasattr(self, '_collision_pairs_logged'):
            self._collision_pairs_logged = set()

        for vehicle in vehicles:
            vid = getattr(vehicle, 'id', None)
            if vid is None:
                # Fallback to object id which may change upon respawn
                vid = id(vehicle)
            crashed = bool(getattr(vehicle, 'crashed', False))
            prev = self._vehicle_crash_state.get(vid, False)
            if crashed and not prev:
                self.collisions += 1
                collisions_this_step += 1
                # Estimate collision severity via relative speed squared with nearest neighbor
                try:
                    pos = np.array(getattr(vehicle, 'position', (0.0, 0.0)), dtype=float)
                    v_speed = float(getattr(vehicle, 'speed', 0.0))
                    v_heading = float(getattr(vehicle, 'heading', 0.0)) if hasattr(vehicle, 'heading') else 0.0
                    v_vel = np.array([v_speed * np.cos(v_heading), v_speed * np.sin(v_heading)])
                    # Find nearest other vehicle within small radius
                    nearest = None
                    nearest_vid = None
                    nearest_dist = float('inf')
                    for other in vehicles:
                        if other is vehicle:
                            continue
                        o_vid = getattr(other, 'id', None) or id(other)
                        o_pos = np.array(getattr(other, 'position', (np.inf, np.inf)), dtype=float)
                        d = float(np.linalg.norm(o_pos - pos))
                        if d < nearest_dist:
                            nearest = other
                            nearest_vid = o_vid
                            nearest_dist = d
                    if nearest is not None and np.isfinite(nearest_dist) and nearest_dist < 5.0:
                        o_speed = float(getattr(nearest, 'speed', 0.0))
                        o_heading = float(getattr(nearest, 'heading', 0.0)) if hasattr(nearest, 'heading') else 0.0
                        o_vel = np.array([o_speed * np.cos(o_heading), o_speed * np.sin(o_heading)])
                        rel_v = o_vel - v_vel
                        css = float(np.dot(rel_v, rel_v))  # v_rel^2
                        pair = tuple(sorted([vid, nearest_vid]))
                        if pair not in getattr(self, '_collision_pairs_logged', set()):
                            self.collision_events.append({
                                'step': int(step_num),
                                'vid_a': vid,
                                'vid_b': nearest_vid,
                                'relative_speed_sq': css,
                                'distance': float(nearest_dist)
                            })
                            self._collision_pairs_logged.add(pair)
                except Exception:
                    pass
            self._vehicle_crash_state[vid] = crashed

        if self.step_data:
            self.step_data[-1]['collisions_this_step'] = collisions_this_step

    def _lane_and_s(self, road, vehicle):
        lane = None
        s = None
        try:
            net = getattr(road, 'network', None)
            lane_index = getattr(vehicle, 'lane_index', None)
            if net is not None and lane_index is not None:
                lane = net.get_lane(lane_index)
                if lane is not None and hasattr(lane, 'local_coordinates') and hasattr(vehicle, 'position'):
                    s, _ = lane.local_coordinates(vehicle.position)
        except Exception:
            lane = None
            s = None
        return lane, s

    def _front_rear_in_lane(self, road, self_vehicle, lane_index, s_self, self_vid):
        front = None
        rear = None
        front_s = float('inf')
        rear_s = -float('inf')
        try:
            # Prefer highway-env helper if available
            get_nb = getattr(road, 'neighbouring_vehicles', None)
            net = getattr(road, 'network', None)
            lane = net.get_lane(lane_index) if (net is not None and lane_index is not None) else None
            if callable(get_nb):
                try:
                    nb_front, nb_rear = get_nb(self_vehicle)
                    if nb_front is not None and lane is not None and hasattr(lane, 'local_coordinates'):
                        s_f, _ = lane.local_coordinates(nb_front.position)
                        if s_f > s_self:
                            front = (nb_front, s_f)
                    if nb_rear is not None and lane is not None and hasattr(lane, 'local_coordinates'):
                        s_r, _ = lane.local_coordinates(nb_rear.position)
                        if s_r < s_self:
                            rear = (nb_rear, s_r)
                    return front, rear
                except Exception:
                    # Fallback to manual scan
                    pass
            vehicles = getattr(road, 'vehicles', [])
            for v in vehicles:
                vid = getattr(v, 'id', None) or id(v)
                if vid == self_vid:
                    continue
                if getattr(v, 'lane_index', None) != lane_index:
                    continue
                if lane is not None and hasattr(lane, 'local_coordinates') and hasattr(v, 'position'):
                    try:
                        s_v, _ = lane.local_coordinates(v.position)
                    except Exception:
                        continue
                    if s_v > s_self and s_v < front_s:
                        front_s = s_v
                        front = (v, s_v)
                    if s_v < s_self and s_v > rear_s:
                        rear_s = s_v
                        rear = (v, s_v)
        except Exception:
            return None, None
        return front, rear

    def _calculate_neighbor_safety_events(self, road, vehicles):
        ttc_list = []
        try:
            for v in vehicles:
                if getattr(v, 'crashed', False):
                    continue
                vid = getattr(v, 'id', None) or id(v)
                lane, s = self._lane_and_s(road, v)
                if lane is None or s is None:
                    continue
                front, rear = self._front_rear_in_lane(road, v, getattr(v, 'lane_index', None), s, vid)
                # Headway / DRAC / TTC with front vehicle
                if front is not None:
                    front_v, s_front = front
                    gap = float(s_front - s)
                    v_f = float(getattr(v, 'speed', 0.0))
                    v_l = float(getattr(front_v, 'speed', 0.0))
                    # THW
                    if v_f > 1e-3 and gap > 0:
                        thw = gap / v_f
                        self.thw_values.append(thw)
                        if thw < self.thw_violation_threshold:
                            self.thw_violations += 1
                    # DRAC and TTC (longitudinal)
                    dv = v_f - v_l
                    if dv > 0 and gap > 0:
                        ttc = gap / dv
                        if np.isfinite(ttc):
                            ttc_list.append(ttc)
                            if ttc < self.TTC_THRESHOLD:
                                self.ttc_events.append(ttc)
                        drac = (dv * dv) / (2.0 * gap)
                        if np.isfinite(drac):
                            self.drac_values.append(drac)
            if ttc_list:
                self.min_ttc_per_step.append(float(np.min(ttc_list)))
            else:
                self.min_ttc_per_step.append(float('inf'))
        except Exception:
            # Robust to API variations; in worst case, keep prior behavior (no events)
            pass

    def _update_throughput(self, road, vehicles):
        try:
            net = getattr(road, 'network', None)
            for v in vehicles:
                if getattr(v, 'crashed', False):
                    continue
                vid = getattr(v, 'id', None) or id(v)
                lane_index = getattr(v, 'lane_index', None)
                if net is None or lane_index is None:
                    continue
                lane = None
                try:
                    lane = net.get_lane(lane_index)
                except Exception:
                    lane = None
                if lane is None or not hasattr(lane, 'local_coordinates'):
                    continue
                try:
                    s, _ = lane.local_coordinates(v.position)
                except Exception:
                    continue
                last_s = self._last_s_by_vid.get(vid, None)
                lane_length = getattr(lane, 'length', None)
                # If lane length is not available, skip counting for this vehicle to avoid bias
                if lane_length is None:
                    self._last_s_by_vid[vid] = s
                    continue
                threshold = float(lane_length) - float(self._lane_end_threshold)
                if last_s is not None:
                    # Crossing near the end of lane from below to above threshold
                    if last_s < threshold <= s:
                        self.completed_routes += 1
                    # Reset if respawn (s jumps backward a lot)
                    if s + 20.0 < last_s:
                        self._last_s_by_vid[vid] = s
                    else:
                        self._last_s_by_vid[vid] = s
                else:
                    self._last_s_by_vid[vid] = s
        except Exception:
            pass

    def _update_lane_change_quality(self, road, vehicles):
        try:
            net = getattr(road, 'network', None)
            for v in vehicles:
                if getattr(v, 'crashed', False):
                    continue
                vid = getattr(v, 'id', None) or id(v)
                lane_index = getattr(v, 'lane_index', None)
                prev_lane_index = self._prev_lane_index.get(vid, lane_index)
                self._prev_lane_index[vid] = lane_index
                if lane_index is None or prev_lane_index is None:
                    continue
                if lane_index == prev_lane_index:
                    continue
                # Lane change detected
                self.lane_change_attempts += 1
                lane_now, s_now = self._lane_and_s(road, v)
                if lane_now is None or s_now is None:
                    continue
                rear = self._front_rear_in_lane(road, v, lane_index, s_now, vid)[1]
                unsafe = False
                safe_gap = max(5.0, float(getattr(v, 'speed', 0.0)) * 1.0)
                if rear is not None:
                    rear_v, s_rear = rear
                    gap_rear = float(s_now - s_rear)
                    if gap_rear < safe_gap:
                        unsafe = True
                    else:
                        dv = abs(float(getattr(v, 'speed', 0.0)) - float(getattr(rear_v, 'speed', 0.0)))
                        if dv > 5.0:
                            unsafe = True
                else:
                    # No rear car: likely safe
                    unsafe = False
                if unsafe:
                    self.unsafe_lane_changes += 1
        except Exception:
            pass

    def _update_right_lane_compliance(self, road, ego_vehicle):
        try:
            if ego_vehicle is None:
                return
            lane_index = getattr(ego_vehicle, 'lane_index', None)
            net = getattr(road, 'network', None)
            if lane_index is None or net is None:
                return
            # Rightmost is typically lane_id == 0
            lane_id = lane_index[2] if isinstance(lane_index, (list, tuple)) and len(lane_index) > 2 else None
            if lane_id is None:
                return
            # Determine eligibility to keep right: if there exists a right lane and it is sufficiently free
            eligible = False
            compliant = lane_id == 0
            if lane_id > 0:
                right_index = (lane_index[0], lane_index[1], lane_id - 1)
                try:
                    right_lane = net.get_lane(right_index)
                except Exception:
                    right_lane = None
                if right_lane is not None:
                    # Compute front neighbor gap in the right lane
                    lane_now, s_now = self._lane_and_s(road, ego_vehicle)
                    if lane_now is None or s_now is None:
                        return
                    # We need s in right lane coordinates too
                    try:
                        s_right, _ = right_lane.local_coordinates(ego_vehicle.position)
                    except Exception:
                        s_right = s_now
                    front_right, rear_right = self._front_rear_in_lane(road, ego_vehicle, right_index, s_right, getattr(ego_vehicle, 'id', None) or id(ego_vehicle))
                    safe_gap = max(10.0, float(getattr(ego_vehicle, 'speed', 0.0)) * 2.0)
                    gap_ok = True
                    if front_right is not None:
                        gap = float(front_right[1] - s_right)
                        gap_ok = gap > safe_gap
                    # Eligible to be on right if right lane exists and has enough front gap
                    eligible = gap_ok
            if eligible:
                self.right_lane_eligible_ticks += 1
                if compliant:
                    self.right_lane_compliant_ticks += 1
        except Exception:
            pass
    
    def _check_merge_events(self, env, vehicles):
        """
        Check for merge attempts and successes (merge scenarios only).
        
        Args:
            env: Highway environment
            vehicles (list): List of vehicles
        """
        # Use scenario context when available to limit to merge scenario
        scenario_type = None
        try:
            runtime = (self.config or {}).get('runtime', {}) if isinstance(self.config, dict) else {}
            scenario_type = (runtime.get('scenario_type') or '').strip().lower()
        except Exception:
            scenario_type = None
        if scenario_type != 'merge':
            return
        # Simplified merge detection (heuristic)
        try:
            # Count vehicles that successfully changed from merge lane to main highway
            # This is a simplified heuristic - in practice, you'd track specific lane changes
            for vehicle in vehicles:
                if hasattr(vehicle, 'previous_lane_index') and hasattr(vehicle, 'lane_index'):
                    if (vehicle.previous_lane_index != vehicle.lane_index and
                        self._is_merge_lane_change(vehicle.previous_lane_index, vehicle.lane_index)):
                        self.merge_attempts += 1
                        # Assume successful if vehicle made the change without crashing
                        if not getattr(vehicle, 'crashed', False):
                            self.successful_merges += 1
        except Exception:
            return
    
    def _is_merge_lane_change(self, from_lane, to_lane):
        """
        Determine if a lane change represents a merge maneuver.
        
        Args:
            from_lane, to_lane: Lane indices
            
        Returns:
            bool: True if this is a merge lane change
        """
        # Simplified heuristic: assume rightmost lane is merge lane
        # and changes from rightmost to other lanes are merges
        return (from_lane[2] == 0 and to_lane[2] > 0)
    
    def calculate_final_metrics(self):
        """
        Calculate final aggregated metrics for the simulation run.
        
        Returns:
            dict: Dictionary of calculated metrics
        """
        metrics = {}
        # Total steps executed in this run
        try:
            metrics['steps'] = int(len(self.step_data))
        except Exception:
            metrics['steps'] = 0
        
        # Efficiency Metrics
        if self.speed_history:
            metrics['avg_speed'] = np.mean(self.speed_history)
            metrics['speed_std'] = np.std(self.speed_history)
        else:
            metrics['avg_speed'] = 0
            metrics['speed_std'] = 0
        
        metrics['throughput'] = self.completed_routes
        # Time-normalized throughput (vehicles per minute)
        try:
            steps = int(len(self.step_data))
            dt = float(self._dt) if self._dt else None
            if not dt:
                # Fallback: attempt to get policy_frequency again from first step env if available
                dt = None
            if dt and steps > 0:
                elapsed_sec = steps * dt
                metrics['throughput_per_min'] = (self.completed_routes / elapsed_sec) * 60.0 if elapsed_sec > 0 else 0.0
            else:
                metrics['throughput_per_min'] = 0.0
        except Exception:
            metrics['throughput_per_min'] = 0.0
        
        # Safety Metrics
        metrics['total_collisions'] = self.collisions
        metrics['ego_collisions'] = int(self.ego_collisions)
        # Event-level TTC (near-miss list)
        if self.ttc_events:
            metrics['avg_ttc'] = float(np.mean(self.ttc_events))
            metrics['min_ttc'] = float(np.min(self.ttc_events))
            metrics['ttc_events_count'] = int(len(self.ttc_events))
        else:
            metrics['avg_ttc'] = float('inf')
            metrics['min_ttc'] = float('inf')
            metrics['ttc_events_count'] = 0
        # Step-level minimum TTC
        if self.min_ttc_per_step:
            finite_step_ttc = [t for t in self.min_ttc_per_step if np.isfinite(t)]
            metrics['min_step_ttc'] = float(np.min(finite_step_ttc)) if finite_step_ttc else float('inf')
            under_1 = sum(1 for t in self.min_ttc_per_step if np.isfinite(t) and t < 1.0)
            under_2 = sum(1 for t in self.min_ttc_per_step if np.isfinite(t) and t < 2.0)
            total_steps = max(1, len(self.min_ttc_per_step))
            metrics['frac_steps_ttc_lt_1'] = under_1 / total_steps
            metrics['frac_steps_ttc_lt_2'] = under_2 / total_steps
        else:
            metrics['min_step_ttc'] = float('inf')
            metrics['frac_steps_ttc_lt_1'] = 0.0
            metrics['frac_steps_ttc_lt_2'] = 0.0
        # THW and DRAC
        if self.thw_values:
            thw_arr = np.asarray(self.thw_values, dtype=float)
            metrics['thw_median'] = float(np.median(thw_arr))
            metrics['thw_p05'] = float(np.percentile(thw_arr, 5))
            metrics['thw_violation_rate'] = float(self.thw_violations) / float(max(1, len(self.thw_values)))
        else:
            metrics['thw_median'] = float('inf')
            metrics['thw_p05'] = float('inf')
            metrics['thw_violation_rate'] = 0.0
        if self.drac_values:
            drac_arr = np.asarray(self.drac_values, dtype=float)
            metrics['drac_mean'] = float(np.mean(drac_arr))
            metrics['drac_p95'] = float(np.percentile(drac_arr, 95))
        else:
            metrics['drac_mean'] = 0.0
            metrics['drac_p95'] = 0.0
        # Collision severity
        if self.collision_events:
            css_arr = np.asarray([e.get('relative_speed_sq', 0.0) for e in self.collision_events], dtype=float)
            metrics['collision_severity_mean'] = float(np.mean(css_arr))
            metrics['collision_severity_p95'] = float(np.percentile(css_arr, 95))
        else:
            metrics['collision_severity_mean'] = 0.0
            metrics['collision_severity_p95'] = 0.0
        
        # Stability Metrics
        all_accelerations = []
        for vehicle_accels in self.acceleration_history.values():
            all_accelerations.extend(vehicle_accels)
        
        if all_accelerations:
            metrics['acceleration_std'] = np.std(all_accelerations)
            metrics['acceleration_mean'] = np.mean(all_accelerations)
        else:
            metrics['acceleration_std'] = 0
            metrics['acceleration_mean'] = 0
        # Jerk
        if self.jerk_values:
            j = np.asarray(self.jerk_values, dtype=float)
            metrics['jerk_mean'] = float(np.mean(j))
            metrics['jerk_p95'] = float(np.percentile(j, 95))
            metrics['jerk_max'] = float(np.max(j))
            exceed = np.sum(np.abs(j) > 2.0)
            metrics['jerk_exceedance_pct_2'] = float(exceed) / float(len(j))
        else:
            metrics['jerk_mean'] = 0.0
            metrics['jerk_p95'] = 0.0
            metrics['jerk_max'] = 0.0
            metrics['jerk_exceedance_pct_2'] = 0.0

        # Lane change quality
        metrics['lane_change_attempts'] = int(self.lane_change_attempts)
        metrics['unsafe_lane_change_rate'] = (
            float(self.unsafe_lane_changes) / float(self.lane_change_attempts)
            if self.lane_change_attempts > 0 else 0.0
        )

        # Right-lane compliance
        metrics['right_lane_compliance_ratio'] = (
            float(self.right_lane_compliant_ticks) / float(max(1, self.right_lane_eligible_ticks))
        )
        
        # Cooperation Metrics (for merge scenarios)
        if self.merge_attempts > 0:
            metrics['merge_success_rate'] = self.successful_merges / self.merge_attempts
        else:
            metrics['merge_success_rate'] = 0
        
        metrics['merge_attempts'] = self.merge_attempts
        metrics['successful_merges'] = self.successful_merges
        
        # Network-wide efficiency metric
        metrics['avg_network_speed'] = (np.mean(self.network_speed_history)
                                         if self.network_speed_history else 0.0)
        
        # Safety normalization: crash rate per 1000 vehicle-steps
        metrics['crash_rate_per_1k_vsteps'] = (
            (self.collisions / self.vehicle_steps) * 1000.0
            if self.vehicle_steps else 0.0
        )
        
        return metrics
    
    def get_step_data_dataframe(self):
        """
        Get step-by-step data as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with step-by-step metrics
        """
        return pd.DataFrame(self.step_data)


class MetricsAggregator:
    """
    Aggregates metrics across multiple simulation runs.
    """
    
    def __init__(self):
        self.all_metrics = []
    
    def add_run_metrics(self, scenario_name, composition_name, run_metrics):
        """
        Add metrics from a single simulation run.
        
        Args:
            scenario_name (str): Name of scenario (e.g., "Highway", "Merge")
            composition_name (str): Agent composition (e.g., "100% Selfish")
            run_metrics (dict): Metrics from single run
        """
        metrics_entry = {
            'scenario': scenario_name,
            'composition': composition_name,
            **run_metrics
        }
        self.all_metrics.append(metrics_entry)
    
    def get_aggregated_results(self):
        """
        Get aggregated results across all runs.
        
        Returns:
            pd.DataFrame: Aggregated metrics grouped by scenario and composition
        """
        if not self.all_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_metrics)
        
        # Group by scenario and composition, calculate mean, std, and count
        agg_map = {
            'avg_speed': ['mean', 'std', 'count'],
            'total_collisions': ['mean', 'std', 'count'],
            'acceleration_std': ['mean', 'std', 'count'],
            'merge_success_rate': ['mean', 'std', 'count'],
            'ttc_events_count': ['mean', 'std', 'count'],
            'throughput': ['mean', 'std', 'count']
        }
        # Include network-wide speed when available
        if 'avg_network_speed' in df.columns:
            agg_map['avg_network_speed'] = ['mean', 'std', 'count']

        aggregated = df.groupby(['scenario', 'composition']).agg(agg_map).round(3)
        
        # Flatten column names
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        
        # Add overall sample size column (should be same for all metrics)
        # Prefer network speed count if present, else fall back to avg_speed_count
        if 'avg_network_speed_count' in aggregated.columns:
            aggregated['sample_size'] = aggregated['avg_network_speed_count']
        else:
            aggregated['sample_size'] = aggregated.get('avg_speed_count', 0)
        
        return aggregated
    
    def save_results(self, output_dir, filename):
        """
        Save aggregated results to file.
        
        Args:
            output_dir (str): Output directory
            filename (str): Output filename
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        results_df = self.get_aggregated_results()
        
        filepath = os.path.join(output_dir, filename)
        results_df.to_csv(filepath, index=False)
        
        return filepath