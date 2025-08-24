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
        # Ego-specific safety
        self.ego_collisions = 0
        self._ego_collision_counted = False
        
        # Stability metrics
        self.acceleration_history = defaultdict(list)
        
        # Cooperation metrics (for merge scenarios)
        self.merge_attempts = 0
        self.successful_merges = 0
        
        # Per-step data for detailed analysis
        self.step_data = []
    
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
        
        # Collect accelerations for stability metrics
        for vehicle in vehicles:
            accel = None
            
            # Try different ways to access acceleration
            if hasattr(vehicle, 'acceleration'):
                accel = vehicle.acceleration
                # Check if it's a method and call it if needed
                if callable(accel):
                    try:
                        accel = accel()
                    except:
                        accel = None
            elif hasattr(vehicle, 'action') and hasattr(vehicle.action, 'acceleration'):
                accel = vehicle.action.acceleration
                if callable(accel):
                    try:
                        accel = accel()
                    except:
                        accel = None
            elif hasattr(vehicle, 'last_action') and hasattr(vehicle.last_action, 'acceleration'):
                accel = vehicle.last_action.acceleration
                if callable(accel):
                    try:
                        accel = accel()
                    except:
                        accel = None
            
            # Only append if we have a valid numeric value
            if accel is not None and not getattr(vehicle, 'crashed', False):
                try:
                    # Ensure it's a number
                    accel_val = float(accel)
                    self.acceleration_history[id(vehicle)].append(accel_val)
                except (TypeError, ValueError):
                    # Skip if we can't convert to float
                    pass
        
        # Check for collisions (transition-based)
        self._check_collisions(vehicles)
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
        
        # Calculate Time-to-Collision events (guarded)
        self._calculate_ttc_events(vehicles)
        
        # Check merge events (for merge scenarios)
        self._check_merge_events(env, vehicles)
        
        # Store step data
        ego_speed = 0
        if ego_vehicle is not None and hasattr(ego_vehicle, 'speed') and not getattr(ego_vehicle, 'crashed', False):
            ego_speed = ego_vehicle.speed
                
        step_data = {
            'step': step_num,
            'num_vehicles': len(vehicles),
            'avg_speed': ego_speed,  # Now tracks only ego vehicle speed
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
    
    def _check_collisions(self, vehicles):
        """
        Check for collisions using highway-env's built-in crash detection.
        
        Args:
            vehicles (list): List of vehicles in simulation
        """
        collisions_this_step = 0
        # Maintain per-vehicle crash state to detect transitions
        if not hasattr(self, '_vehicle_crash_state'):
            self._vehicle_crash_state = {}

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
            self._vehicle_crash_state[vid] = crashed

        if self.step_data:
            self.step_data[-1]['collisions_this_step'] = collisions_this_step
    
    def _calculate_ttc_events(self, vehicles):
        """
        Calculate Time-to-Collision for vehicle pairs.
        
        Args:
            vehicles (list): List of vehicles in simulation
        """
        valid_pairs = 0
        for i, vehicle1 in enumerate(vehicles):
            for vehicle2 in vehicles[i+1:]:
                ttc = self._calculate_ttc_pair(vehicle1, vehicle2)
                if ttc is None:
                    continue
                valid_pairs += 1
                if ttc < self.TTC_THRESHOLD:
                    self.ttc_events.append(ttc)
    
    def _calculate_ttc_pair(self, vehicle1, vehicle2):
        """
        Calculate TTC between two vehicles.
        
        Args:
            vehicle1, vehicle2: Vehicle objects
            
        Returns:
            float or None: TTC in seconds, None if no collision course
        """
        # Guard required attributes
        if not (hasattr(vehicle1, 'position') and hasattr(vehicle2, 'position')):
            return None
        if not (hasattr(vehicle1, 'speed') and hasattr(vehicle2, 'speed')):
            return None
        # Some envs may not expose heading; skip pair if missing
        if not (hasattr(vehicle1, 'heading') and hasattr(vehicle2, 'heading')):
            return None

        try:
            pos1 = np.array(vehicle1.position, dtype=float)
            pos2 = np.array(vehicle2.position, dtype=float)
            v1 = float(getattr(vehicle1, 'speed', 0.0))
            v2 = float(getattr(vehicle2, 'speed', 0.0))
            h1 = float(getattr(vehicle1, 'heading', 0.0))
            h2 = float(getattr(vehicle2, 'heading', 0.0))
        except Exception:
            return None

        vel1 = np.array([v1 * np.cos(h1), v1 * np.sin(h1)])
        vel2 = np.array([v2 * np.cos(h2), v2 * np.sin(h2)])
        
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # If relative velocity is zero or vehicles are moving apart, no collision
        if np.dot(rel_pos, rel_vel) >= 0:
            return None
        
        # Calculate TTC
        rel_speed = np.linalg.norm(rel_vel)
        if rel_speed == 0:
            return None
        
        # Time to closest approach
        t_closest = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)
        
        # Distance at closest approach
        closest_distance = np.linalg.norm(rel_pos + rel_vel * t_closest)
        
        # If closest distance is greater than collision threshold, no collision
        if closest_distance > self.COLLISION_DISTANCE:
            return None
        
        return t_closest
    
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
        
        # Safety Metrics
        metrics['total_collisions'] = self.collisions
        metrics['ego_collisions'] = int(self.ego_collisions)
        if self.ttc_events:
            metrics['avg_ttc'] = np.mean(self.ttc_events)
            metrics['min_ttc'] = np.min(self.ttc_events)
            metrics['ttc_events_count'] = len(self.ttc_events)
        else:
            metrics['avg_ttc'] = float('inf')
            metrics['min_ttc'] = float('inf')  
            metrics['ttc_events_count'] = 0
        
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