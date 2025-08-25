"""
Roundabout-Specific Policy Implementation

This module implements enhanced cooperative behaviors specifically designed
for roundabout scenarios, extending the base cooperative policy with
roundabout-specific social laws.
"""

import numpy as np
import logging
import math
from .cooperative_policy import CooperativePolicy


class RoundaboutCooperativePolicy(CooperativePolicy):
    """
    Enhanced cooperative policy for roundabout scenarios.
    
    Implements roundabout-specific social laws:
    - Entry Facilitation: Helps vehicles enter the roundabout safely
    - Smooth Flow Maintenance: Maintains consistent spacing and speed
    - Exit Courtesy: Assists vehicles exiting the roundabout
    """
    
    def __init__(self, config=None):
        """Initialize roundabout-specific cooperative policy."""
        super().__init__(config)
        
        # Roundabout-specific social law parameters
        if config and 'social_laws' in config:
            social_laws = config['social_laws']
            
            # Entry Facilitation parameters
            if 'entry_facilitation' in social_laws:
                entry_config = social_laws['entry_facilitation']
                self.entry_detection_distance = entry_config.get('entry_detection_distance', 50.0)
                self.facilitation_speed_factor = entry_config.get('facilitation_speed_factor', 0.9)
                self.gap_creation_distance = entry_config.get('gap_creation_distance', 15.0)
                self.entry_facilitation_enabled = entry_config.get('enabled', True)
            else:
                self.entry_detection_distance = 50.0
                self.facilitation_speed_factor = 0.9
                self.gap_creation_distance = 15.0
                self.entry_facilitation_enabled = True
                
            # Smooth Flow Maintenance parameters
            if 'smooth_flow_maintenance' in social_laws:
                flow_config = social_laws['smooth_flow_maintenance']
                self.target_spacing_factor = flow_config.get('target_spacing_factor', 1.8)
                self.speed_harmonization_rate = flow_config.get('speed_harmonization_rate', 0.1)
                self.accordion_prevention_threshold = flow_config.get('accordion_prevention_threshold', 5.0)
                self.flow_maintenance_enabled = flow_config.get('enabled', True)
            else:
                self.target_spacing_factor = 1.8
                self.speed_harmonization_rate = 0.1
                self.accordion_prevention_threshold = 5.0
                self.flow_maintenance_enabled = True
                
            # Exit Courtesy parameters
            if 'exit_courtesy' in social_laws:
                exit_config = social_laws['exit_courtesy']
                self.early_signal_distance = exit_config.get('early_signal_distance', 30.0)
                self.exit_gap_provision = exit_config.get('exit_gap_provision', True)
                self.lane_change_assistance = exit_config.get('lane_change_assistance', True)
                self.exit_courtesy_enabled = exit_config.get('enabled', True)
            else:
                self.early_signal_distance = 30.0
                self.exit_gap_provision = True
                self.lane_change_assistance = True
                self.exit_courtesy_enabled = True
        else:
            # Default parameters
            self.entry_detection_distance = 50.0
            self.facilitation_speed_factor = 0.9
            self.gap_creation_distance = 15.0
            self.entry_facilitation_enabled = True
            self.target_spacing_factor = 1.8
            self.speed_harmonization_rate = 0.1
            self.accordion_prevention_threshold = 5.0
            self.flow_maintenance_enabled = True
            self.early_signal_distance = 30.0
            self.exit_gap_provision = True
            self.lane_change_assistance = True
            self.exit_courtesy_enabled = True
        
        # Roundabout geometry parameters
        if config and 'environment' in config and 'roundabout' in config['environment']:
            env_config = config['environment']['roundabout']
            self.roundabout_radius = env_config.get('radius', 50)
            self.entry_points = env_config.get('entry_points', 4)
        else:
            self.roundabout_radius = 50
            self.entry_points = 4
        
        # Internal state for roundabout navigation
        self.current_angle = 0.0
        self.target_exit_angle = None
        self.entry_assistance_cooldown = 0
        
        logging.info("RoundaboutCooperativePolicy initialized with roundabout-specific social laws")
    
    def act(self, obs):
        """
        Enhanced action selection with roundabout-specific cooperative behaviors.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return super().act(obs)
        
        # Analyze roundabout context
        roundabout_context = self._analyze_roundabout_context(obs)
        
        # Update internal roundabout state
        self._update_roundabout_position(roundabout_context)
        
        # Check for roundabout-specific social law opportunities
        if self._should_facilitate_entry(roundabout_context):
            return self._execute_entry_facilitation(obs)
        
        if self._should_maintain_smooth_flow(roundabout_context):
            return self._execute_flow_maintenance(obs)
        
        if self._should_provide_exit_courtesy(roundabout_context):
            return self._execute_exit_courtesy(obs)
        
        # Apply roundabout-specific navigation logic
        return self._navigate_roundabout(obs, roundabout_context)
    
    def _analyze_roundabout_context(self, obs):
        """
        Analyze the roundabout traffic context.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            dict: Context information about roundabout traffic
        """
        context = {
            'ego_position': None,
            'ego_angle': 0.0,
            'entering_vehicles': [],
            'exiting_vehicles': [],
            'circulating_vehicles': [],
            'flow_consistency': 1.0,
            'in_roundabout': False
        }
        
        if len(obs) == 0:
            return context
        
        # Ego vehicle information
        ego_info = obs[0]
        ego_x = ego_info[1] if len(ego_info) > 1 else 0
        ego_y = ego_info[2] if len(ego_info) > 2 else 0
        
        context['ego_position'] = {'x': ego_x, 'y': ego_y}
        
        # Calculate ego angle in roundabout (polar coordinates)
        ego_distance_from_center = math.sqrt(ego_x**2 + ego_y**2)
        context['in_roundabout'] = ego_distance_from_center < self.roundabout_radius * 1.5
        
        if context['in_roundabout']:
            context['ego_angle'] = math.atan2(ego_y, ego_x)
        
        # Analyze other vehicles
        speeds = []
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                vx = vehicle[1]
                vy = vehicle[2]
                vehicle_x = ego_x + vx  # Absolute position approximation
                vehicle_y = ego_y + vy
                
                vehicle_distance_from_center = math.sqrt(vehicle_x**2 + vehicle_y**2)
                vehicle_speed = math.sqrt(vehicle[3]**2 + vehicle[4]**2) if len(vehicle) > 4 else 20
                speeds.append(vehicle_speed)
                
                vehicle_info = {
                    'index': i,
                    'rel_x': vx,
                    'rel_y': vy,
                    'speed': vehicle_speed,
                    'distance_from_center': vehicle_distance_from_center
                }
                
                # Categorize vehicles based on position and movement
                if vehicle_distance_from_center > self.roundabout_radius * 1.2:
                    # Outside roundabout - potentially entering
                    if abs(vx) > 5:  # Moving toward roundabout
                        context['entering_vehicles'].append(vehicle_info)
                elif vehicle_distance_from_center < self.roundabout_radius * 0.8:
                    # Inside roundabout - circulating
                    context['circulating_vehicles'].append(vehicle_info)
                    # Check if exiting (moving away from center)
                    if vx * vehicle_x + vy * vehicle_y > 0:  # Dot product positive = moving away
                        context['exiting_vehicles'].append(vehicle_info)
        
        # Calculate flow consistency
        if speeds:
            avg_speed = np.mean(speeds)
            speed_variance = np.var(speeds)
            context['flow_consistency'] = max(0.0, 1.0 - speed_variance / (avg_speed + 1e-6))
        
        return context
    
    def _should_facilitate_entry(self, context):
        """
        Determine if we should facilitate roundabout entry for other vehicles.
        
        Args:
            context: Roundabout context
            
        Returns:
            bool: True if entry facilitation is beneficial
        """
        if not self.entry_facilitation_enabled or not context['in_roundabout']:
            return False
        
        if self.entry_assistance_cooldown > 0:
            self.entry_assistance_cooldown -= 1
            return False
        
        # Check for vehicles trying to enter
        for entering_vehicle in context['entering_vehicles']:
            distance = math.sqrt(entering_vehicle['rel_x']**2 + entering_vehicle['rel_y']**2)
            if distance < self.entry_detection_distance:
                # Check if we're in a position to help (ahead of entry point)
                if entering_vehicle['rel_x'] < 0:  # Vehicle is behind us
                    return True
        
        return False
    
    def _should_maintain_smooth_flow(self, context):
        """
        Determine if flow maintenance is needed.
        
        Args:
            context: Roundabout context
            
        Returns:
            bool: True if flow maintenance is beneficial
        """
        if not self.flow_maintenance_enabled or not context['in_roundabout']:
            return False
        
        # Check flow consistency
        return context['flow_consistency'] < 0.7  # Below threshold
    
    def _should_provide_exit_courtesy(self, context):
        """
        Determine if exit courtesy should be provided.
        
        Args:
            context: Roundabout context
            
        Returns:
            bool: True if exit courtesy is beneficial
        """
        if not self.exit_courtesy_enabled:
            return False
        
        # Check for vehicles trying to exit
        for exiting_vehicle in context['exiting_vehicles']:
            distance = math.sqrt(exiting_vehicle['rel_x']**2 + exiting_vehicle['rel_y']**2)
            if distance < self.early_signal_distance:
                return True
        
        return False
    
    def _execute_entry_facilitation(self, obs):
        """
        Execute entry facilitation by creating gaps.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action to facilitate entry
        """
        # Slow down to create gap for entering vehicle
        self.entry_assistance_cooldown = 10  # Cooldown to prevent constant assistance
        logging.debug("Executing entry facilitation for roundabout entry")
        return 4  # SLOWER
    
    def _execute_flow_maintenance(self, obs):
        """
        Execute smooth flow maintenance.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action to maintain flow
        """
        # Analyze front vehicle for spacing adjustment
        if len(obs) > 1:
            front_vehicle = self._find_front_vehicle_roundabout(obs)
            if front_vehicle is not None:
                distance = abs(front_vehicle[1])  # Relative x distance
                target_distance = self.target_spacing_factor * 10  # Target spacing
                
                if distance < target_distance * 0.8:
                    logging.debug("Executing flow maintenance - increasing spacing")
                    return 4  # SLOWER to increase spacing
                elif distance > target_distance * 1.5:
                    logging.debug("Executing flow maintenance - decreasing spacing")
                    return 3  # FASTER to decrease spacing
        
        return 1  # IDLE - maintain current flow
    
    def _execute_exit_courtesy(self, obs):
        """
        Execute exit courtesy behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action for exit courtesy
        """
        # Provide gap for exiting vehicles
        logging.debug("Executing exit courtesy for exiting vehicle")
        return 4  # SLOWER to provide gap
    
    def _navigate_roundabout(self, obs, context):
        """
        Navigate the roundabout with cooperative considerations.
        
        Args:
            obs: Highway-env observation array
            context: Roundabout context
            
        Returns:
            int: Navigation action
        """
        if not context['in_roundabout']:
            # Not in roundabout yet - use base policy
            return super().act(obs)
        
        # In roundabout - maintain steady progress
        front_vehicle = self._find_front_vehicle_roundabout(obs)
        
        if front_vehicle is not None:
            distance = abs(front_vehicle[1])
            if distance < 15:  # Too close
                return 4  # SLOWER
            elif distance > 30:  # Can go faster
                return 3  # FASTER
        else:
            # No vehicle ahead - maintain steady speed
            ego_speed = obs[0][3] if len(obs[0]) > 3 else 20
            if ego_speed < 20:  # Too slow for roundabout
                return 3  # FASTER
        
        return 1  # IDLE - steady progress
    
    def _find_front_vehicle_roundabout(self, obs):
        """
        Find the front vehicle in roundabout context.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            array or None: Front vehicle observation
        """
        front_vehicles = []
        ego_angle = self.current_angle
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                rel_x, rel_y = vehicle[1], vehicle[2]
                if rel_x > 0:  # Vehicle ahead
                    # In roundabout, "ahead" is more complex due to circular nature
                    distance = math.sqrt(rel_x**2 + rel_y**2)
                    front_vehicles.append((distance, vehicle))
        
        if front_vehicles:
            return min(front_vehicles, key=lambda x: x[0])[1]
        return None
    
    def _update_roundabout_position(self, context):
        """
        Update internal position tracking in roundabout.
        
        Args:
            context: Roundabout context
        """
        if context['in_roundabout']:
            self.current_angle = context['ego_angle']


class RoundaboutSelfishPolicy:
    """
    Selfish policy specialized for roundabout scenarios.
    
    Maintains selfish characteristics while adapting to roundabout geometry.
    """
    
    def __init__(self, config=None):
        """Initialize roundabout-specific selfish policy."""
        from .selfish_policy import SelfishPolicy
        self.base_policy = SelfishPolicy(config)
        
        # Roundabout-specific parameters (more aggressive entry/exit)
        self.aggressive_entry = True
        self.gap_acceptance_threshold = 0.4  # Lower threshold for entry gaps
        self.exit_speed_maintenance = True   # Maintain speed during exit
        
        logging.info("RoundaboutSelfishPolicy initialized")
    
    def act(self, obs):
        """
        Action selection with roundabout-specific selfish optimizations.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        # Use base selfish policy with roundabout adjustments
        base_action = self.base_policy.act(obs)
        
        # Be more aggressive about maintaining speed in roundabout
        if self._is_in_roundabout(obs):
            if base_action == 4:  # Don't slow down as much in roundabout
                return 1  # IDLE instead of SLOWER for more aggressive flow
        
        return base_action
    
    def _is_in_roundabout(self, obs):
        """
        Detect if currently in roundabout.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            bool: True if in roundabout
        """
        if len(obs) == 0:
            return False
        
        # Simple heuristic based on position
        ego_info = obs[0]
        ego_x = ego_info[1] if len(ego_info) > 1 else 0
        ego_y = ego_info[2] if len(ego_info) > 2 else 0
        
        # Assume in roundabout if within certain radius of center
        distance_from_center = math.sqrt(ego_x**2 + ego_y**2)
        return distance_from_center < 60  # Within roundabout area