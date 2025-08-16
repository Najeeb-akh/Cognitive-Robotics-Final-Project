"""
Extended Metrics Collection System

This module extends the original metrics collection with scenario-specific
metrics for intersection, roundabout, and racetrack scenarios while preserving
all original functionality.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import math
import logging

# Import original metrics system (unchanged)
from metrics import MetricsCollector as OriginalMetricsCollector, MetricsAggregator


class ExtendedMetricsCollector(OriginalMetricsCollector):
    """
    Extended metrics collector that adds scenario-specific metrics
    while preserving all original functionality.
    """
    
    def __init__(self, config, scenario_type='highway'):
        """
        Initialize extended metrics collector.
        
        Args:
            config (dict): Configuration dictionary
            scenario_type (str): Type of scenario for specialized metrics
        """
        # Initialize original metrics collector (preserves all functionality)
        super().__init__(config)
        
        self.scenario_type = scenario_type
        
        # Extended metrics configuration
        extended_metrics_config = config.get('metrics', {})
        
        # Intersection-specific metrics configuration
        if scenario_type == 'intersection':
            intersection_config = extended_metrics_config.get('intersection', {})
            self.track_turn_success = intersection_config.get('turn_success_rate', True)
            self.track_waiting_time = intersection_config.get('waiting_time_analysis', True)
            self.track_throughput = intersection_config.get('throughput_measurement', True)
            self.track_conflict_resolution = intersection_config.get('conflict_resolution_time', True)
            
        # Roundabout-specific metrics configuration
        elif scenario_type == 'roundabout':
            roundabout_config = extended_metrics_config.get('roundabout', {})
            self.track_entry_efficiency = roundabout_config.get('entry_efficiency', True)
            self.track_flow_optimization = roundabout_config.get('flow_optimization', True)
            self.track_lane_utilization = roundabout_config.get('lane_utilization', True)
            self.track_yield_compliance = roundabout_config.get('yield_compliance', True)
            
        # Racetrack-specific metrics configuration
        elif scenario_type == 'racetrack':
            racetrack_config = extended_metrics_config.get('racetrack', {})
            self.track_overtaking_success = racetrack_config.get('overtaking_success', True)
            self.track_speed_efficiency = racetrack_config.get('speed_efficiency', True)
            self.track_cooperation = racetrack_config.get('cooperation_tracking', True)
            self.track_high_speed_safety = racetrack_config.get('high_speed_safety', True)
        
        # Reset extended metrics
        self.reset_extended_metrics()
        
        logging.info(f"ExtendedMetricsCollector initialized for {scenario_type} scenario")
    
    def reset_metrics(self):
        """Reset all metrics including original and extended ones."""
        # Reset original metrics
        super().reset_metrics()
        
        # Reset extended metrics
        self.reset_extended_metrics()
    
    def reset_extended_metrics(self):
        """Reset extended scenario-specific metrics."""
        # Intersection metrics
        if self.scenario_type == 'intersection':
            self.turn_attempts = 0
            self.successful_turns = 0
            self.waiting_times = []
            self.vehicles_through_intersection = 0
            self.conflict_resolution_times = []
            self.total_intersection_time = 0
            
        # Roundabout metrics  
        elif self.scenario_type == 'roundabout':
            self.entry_attempts = 0
            self.successful_entries = 0
            self.entry_waiting_times = []
            self.roundabout_throughput = 0
            self.inner_lane_usage = 0
            self.outer_lane_usage = 0
            self.yield_events = 0
            self.proper_yields = 0
            
        # Racetrack metrics
        elif self.scenario_type == 'racetrack':
            self.overtaking_attempts = 0
            self.successful_overtakes = 0
            self.failed_overtakes = 0
            self.lap_times = []
            self.slipstream_events = 0
            self.cooperation_events = 0
            self.high_speed_incidents = 0
            self.max_speed_achieved = 0
    
    def collect_step_metrics(self, env, step):
        """
        Collect metrics for current step including original and extended metrics.
        
        Args:
            env: Highway environment
            step (int): Current simulation step
        """
        # Collect original metrics (unchanged)
        super().collect_step_metrics(env, step)
        
        # Collect extended scenario-specific metrics
        self._collect_extended_step_metrics(env, step)
    
    def _collect_extended_step_metrics(self, env, step):
        """
        Collect scenario-specific metrics for current step.
        
        Args:
            env: Highway environment  
            step (int): Current simulation step
        """
        if self.scenario_type == 'intersection':
            self._collect_intersection_metrics(env, step)
        elif self.scenario_type == 'roundabout':
            self._collect_roundabout_metrics(env, step)
        elif self.scenario_type == 'racetrack':
            self._collect_racetrack_metrics(env, step)
    
    def _collect_intersection_metrics(self, env, step):
        """Collect intersection-specific metrics."""
        try:
            # Get environment state
            vehicles = getattr(env.unwrapped, 'controlled_vehicles', []) + getattr(env.unwrapped, 'road', {}).get('vehicles', [])
            
            if not vehicles:
                return
            
            # Track vehicles passing through intersection
            intersection_center = (0, 0)  # Simplified intersection center
            intersection_radius = 25
            
            for vehicle in vehicles:
                if hasattr(vehicle, 'position'):
                    distance_to_center = np.sqrt(vehicle.position[0]**2 + vehicle.position[1]**2)
                    
                    # Count vehicles passing through intersection
                    if distance_to_center < intersection_radius:
                        self.vehicles_through_intersection += 1
                    
                    # Detect turning behavior (simplified)
                    if hasattr(vehicle, 'velocity') and len(vehicle.velocity) > 1:
                        lateral_velocity = abs(vehicle.velocity[1])
                        if lateral_velocity > 2.0:  # Turning threshold
                            self.turn_attempts += 1
                            # Success if vehicle maintains control (no collision)
                            if not getattr(vehicle, 'crashed', False):
                                self.successful_turns += 1
                    
                    # Track waiting times (vehicles with very low speed near intersection)
                    if distance_to_center < intersection_radius * 1.5:
                        speed = np.linalg.norm(vehicle.velocity) if hasattr(vehicle, 'velocity') else 0
                        if speed < 2.0:  # Nearly stopped
                            self.waiting_times.append(1)  # 1 step of waiting
            
            self.total_intersection_time += 1
            
        except Exception as e:
            logging.debug(f"Error collecting intersection metrics: {e}")
    
    def _collect_roundabout_metrics(self, env, step):
        """Collect roundabout-specific metrics."""
        try:
            # Get environment state
            vehicles = getattr(env.unwrapped, 'controlled_vehicles', []) + getattr(env.unwrapped, 'road', {}).get('vehicles', [])
            
            if not vehicles:
                return
            
            # Roundabout geometry (simplified)
            roundabout_center = (0, 0)
            roundabout_radius = 50
            entry_zone_radius = 70
            
            for vehicle in vehicles:
                if hasattr(vehicle, 'position'):
                    distance_to_center = np.sqrt(vehicle.position[0]**2 + vehicle.position[1]**2)
                    
                    # Track entry attempts and success
                    if entry_zone_radius > distance_to_center > roundabout_radius:
                        # In entry zone
                        if hasattr(vehicle, 'velocity'):
                            speed = np.linalg.norm(vehicle.velocity)
                            if speed > 1.0:  # Moving toward roundabout
                                self.entry_attempts += 1
                                if distance_to_center < roundabout_radius * 1.1:  # Successfully entering
                                    self.successful_entries += 1
                            else:  # Waiting to enter
                                self.entry_waiting_times.append(1)
                    
                    # Track lane utilization in roundabout
                    elif distance_to_center < roundabout_radius:
                        # Inside roundabout
                        self.roundabout_throughput += 1
                        if hasattr(vehicle, 'lane_index'):
                            if vehicle.lane_index == 0:  # Inner lane
                                self.inner_lane_usage += 1
                            else:  # Outer lane
                                self.outer_lane_usage += 1
                    
                    # Track yielding behavior (simplified)
                    if hasattr(vehicle, 'velocity'):
                        speed = np.linalg.norm(vehicle.velocity)
                        if speed < 1.0 and entry_zone_radius > distance_to_center > roundabout_radius:
                            self.yield_events += 1
                            # Proper yield if no collision follows
                            if not getattr(vehicle, 'crashed', False):
                                self.proper_yields += 1
            
        except Exception as e:
            logging.debug(f"Error collecting roundabout metrics: {e}")
    
    def _collect_racetrack_metrics(self, env, step):
        """Collect racetrack-specific metrics."""
        try:
            # Get environment state
            vehicles = getattr(env.unwrapped, 'controlled_vehicles', []) + getattr(env.unwrapped, 'road', {}).get('vehicles', [])
            
            if not vehicles:
                return
            
            for vehicle in vehicles:
                if hasattr(vehicle, 'velocity'):
                    speed = np.linalg.norm(vehicle.velocity)
                    self.max_speed_achieved = max(self.max_speed_achieved, speed)
                    
                    # Track high-speed behavior (> 30 m/s)
                    if speed > 30:
                        # Look for nearby vehicles to detect overtaking
                        for other in vehicles:
                            if other != vehicle and hasattr(other, 'position'):
                                distance = np.linalg.norm(np.array(vehicle.position) - np.array(other.position))
                                if distance < 20:  # Close proximity
                                    relative_speed = speed - np.linalg.norm(other.velocity) if hasattr(other, 'velocity') else 0
                                    
                                    if relative_speed > 5:  # Significantly faster
                                        self.overtaking_attempts += 1
                                        # Success if vehicle gets ahead without collision
                                        if not getattr(vehicle, 'crashed', False):
                                            self.successful_overtakes += 1
                                        else:
                                            self.failed_overtakes += 1
                                    
                                    # Detect slipstreaming (following closely at similar speeds)
                                    if 3 < distance < 15 and abs(relative_speed) < 2:
                                        self.slipstream_events += 1
                    
                    # Track cooperation events (simplified)
                    if hasattr(vehicle, 'position'):
                        # Look for cooperative behaviors like gap creation
                        for other in vehicles:
                            if other != vehicle and hasattr(other, 'position'):
                                distance = np.linalg.norm(np.array(vehicle.position) - np.array(other.position))
                                if 10 < distance < 30:  # Moderate distance
                                    # Detect if vehicle is providing space (slower than optimal)
                                    if speed < 25:  # Below typical racing speed
                                        self.cooperation_events += 1
                    
                    # Track high-speed safety incidents
                    if speed > 35 and getattr(vehicle, 'crashed', False):
                        self.high_speed_incidents += 1
            
        except Exception as e:
            logging.debug(f"Error collecting racetrack metrics: {e}")
    
    def calculate_final_metrics(self):
        """
        Calculate final metrics including original and extended metrics.
        
        Returns:
            dict: Complete metrics dictionary
        """
        # Get original metrics
        original_metrics = super().calculate_final_metrics()
        
        # Calculate extended metrics
        extended_metrics = self._calculate_extended_metrics()
        
        # Combine original and extended metrics
        combined_metrics = {**original_metrics, **extended_metrics}
        
        return combined_metrics
    
    def _calculate_extended_metrics(self):
        """Calculate scenario-specific extended metrics."""
        extended_metrics = {}
        
        if self.scenario_type == 'intersection':
            # Intersection-specific metrics
            extended_metrics['turn_success_rate'] = (
                self.successful_turns / max(self.turn_attempts, 1) * 100
            )
            extended_metrics['avg_waiting_time'] = (
                np.mean(self.waiting_times) if self.waiting_times else 0
            )
            extended_metrics['intersection_throughput'] = (
                self.vehicles_through_intersection / max(self.total_intersection_time, 1) * 60  # vehicles/minute
            )
            extended_metrics['conflict_resolution_efficiency'] = (
                len(self.conflict_resolution_times) / max(self.total_intersection_time, 1) * 100
            )
            
        elif self.scenario_type == 'roundabout':
            # Roundabout-specific metrics
            extended_metrics['entry_success_rate'] = (
                self.successful_entries / max(self.entry_attempts, 1) * 100
            )
            extended_metrics['avg_entry_waiting_time'] = (
                np.mean(self.entry_waiting_times) if self.entry_waiting_times else 0
            )
            extended_metrics['roundabout_flow_rate'] = (
                self.roundabout_throughput / max(1000, 1)  # Normalized flow rate
            )
            total_lane_usage = self.inner_lane_usage + self.outer_lane_usage
            extended_metrics['lane_balance_ratio'] = (
                self.inner_lane_usage / max(total_lane_usage, 1) if total_lane_usage > 0 else 0.5
            )
            extended_metrics['yield_compliance_rate'] = (
                self.proper_yields / max(self.yield_events, 1) * 100
            )
            
        elif self.scenario_type == 'racetrack':
            # Racetrack-specific metrics
            total_overtaking = self.successful_overtakes + self.failed_overtakes
            extended_metrics['overtaking_success_rate'] = (
                self.successful_overtakes / max(total_overtaking, 1) * 100
            )
            extended_metrics['max_speed_achieved'] = self.max_speed_achieved
            extended_metrics['avg_lap_time'] = (
                np.mean(self.lap_times) if self.lap_times else 0
            )
            extended_metrics['slipstream_frequency'] = (
                self.slipstream_events / max(1000, 1)  # Events per 1000 steps
            )
            extended_metrics['cooperation_rate'] = (
                self.cooperation_events / max(1000, 1)  # Events per 1000 steps
            )
            extended_metrics['high_speed_safety_score'] = (
                100 - (self.high_speed_incidents / max(total_overtaking, 1) * 100)
            )
        
        return extended_metrics


# Factory function for creating appropriate metrics collector
def create_metrics_collector(config, scenario_type='highway'):
    """
    Factory function to create appropriate metrics collector.
    
    Args:
        config (dict): Configuration dictionary
        scenario_type (str): Scenario type
        
    Returns:
        MetricsCollector: Appropriate metrics collector instance
    """
    if scenario_type in ['intersection', 'roundabout', 'racetrack']:
        return ExtendedMetricsCollector(config, scenario_type)
    else:
        # Use original collector for original scenarios
        return OriginalMetricsCollector(config)


# Re-export original classes for compatibility
__all__ = ['ExtendedMetricsCollector', 'MetricsAggregator', 'create_metrics_collector']