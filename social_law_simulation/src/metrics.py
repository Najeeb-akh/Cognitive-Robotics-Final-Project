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
    
    def reset_metrics(self):
        """Reset all metrics for a new simulation run."""
        # Efficiency metrics
        self.speed_history = []
        self.completed_routes = 0
        self.total_vehicles_spawned = 0
        
        # Safety metrics
        self.collisions = 0
        self.ttc_events = []
        
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
        road = env.unwrapped.road
        vehicles = road.vehicles
        
        if not vehicles:
            return
        
        # Collect speed for efficiency metrics from ego vehicle only (first vehicle)
        if vehicles:
            ego_vehicle = vehicles[0]  # Ego vehicle is always first
            if hasattr(ego_vehicle, 'speed') and not getattr(ego_vehicle, 'crashed', False):
                self.speed_history.append(ego_vehicle.speed)
        
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
        
        # Check for collisions
        self._check_collisions(vehicles)
        
        # Calculate Time-to-Collision events
        self._calculate_ttc_events(vehicles)
        
        # Check merge events (for merge scenarios)
        self._check_merge_events(env, vehicles)
        
        # Store step data
        ego_speed = 0
        if vehicles:
            ego_vehicle = vehicles[0]
            if hasattr(ego_vehicle, 'speed') and not getattr(ego_vehicle, 'crashed', False):
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
        
        for vehicle in vehicles:
            # Use highway-env's built-in crash detection
            if hasattr(vehicle, 'crashed') and vehicle.crashed:
                # Check if we've already counted this collision
                if not hasattr(vehicle, '_collision_counted'):
                    self.collisions += 1
                    collisions_this_step += 1
                    vehicle._collision_counted = True  # Prevent double-counting
        
        # Update step data
        if self.step_data:
            self.step_data[-1]['collisions_this_step'] = collisions_this_step
    
    def _calculate_ttc_events(self, vehicles):
        """
        Calculate Time-to-Collision for vehicle pairs.
        
        Args:
            vehicles (list): List of vehicles in simulation
        """
        for i, vehicle1 in enumerate(vehicles):
            for vehicle2 in vehicles[i+1:]:
                ttc = self._calculate_ttc_pair(vehicle1, vehicle2)
                if ttc is not None and ttc < self.TTC_THRESHOLD:
                    self.ttc_events.append(ttc)
    
    def _calculate_ttc_pair(self, vehicle1, vehicle2):
        """
        Calculate TTC between two vehicles.
        
        Args:
            vehicle1, vehicle2: Vehicle objects
            
        Returns:
            float or None: TTC in seconds, None if no collision course
        """
        # Position and velocity vectors
        pos1 = np.array(vehicle1.position)
        pos2 = np.array(vehicle2.position)
        vel1 = np.array([vehicle1.speed * np.cos(vehicle1.heading), 
                        vehicle1.speed * np.sin(vehicle1.heading)])
        vel2 = np.array([vehicle2.speed * np.cos(vehicle2.heading),
                        vehicle2.speed * np.sin(vehicle2.heading)])
        
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
        # This is a simplified merge detection
        # In a real implementation, you'd need to track specific merge lanes
        if 'merge' in str(type(env.unwrapped)).lower():
            # Count vehicles that successfully changed from merge lane to main highway
            # This is a simplified heuristic - in practice, you'd track specific lane changes
            for vehicle in vehicles:
                if hasattr(vehicle, 'previous_lane_index') and hasattr(vehicle, 'lane_index'):
                    if (vehicle.previous_lane_index != vehicle.lane_index and
                        self._is_merge_lane_change(vehicle.previous_lane_index, vehicle.lane_index)):
                        self.merge_attempts += 1
                        # Assume successful if vehicle made the change
                        self.successful_merges += 1
    
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
        
        # Group by scenario and composition, calculate mean and std
        aggregated = df.groupby(['scenario', 'composition']).agg({
            'avg_speed': ['mean', 'std'],
            'total_collisions': ['mean', 'std'], 
            'acceleration_std': ['mean', 'std'],
            'merge_success_rate': ['mean', 'std'],
            'ttc_events_count': ['mean', 'std'],
            'throughput': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        aggregated.columns = [f"{col[0]}_{col[1]}" for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        
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