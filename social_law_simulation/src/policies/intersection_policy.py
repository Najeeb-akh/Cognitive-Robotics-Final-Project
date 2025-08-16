"""
Intersection-Specific Policy Implementation

This module implements enhanced cooperative behaviors specifically designed
for intersection scenarios, extending the base cooperative policy with
intersection-specific social laws.
"""

import numpy as np
import logging
from .cooperative_policy import CooperativePolicy


class IntersectionCooperativePolicy(CooperativePolicy):
    """
    Enhanced cooperative policy for intersection scenarios.
    
    Implements intersection-specific social laws:
    - Polite Gap Provision: Creates gaps for turning vehicles
    - Cooperative Turn-Taking: Balances through traffic with turns
    - Adaptive Right-of-Way: Extends courtesy based on waiting times
    """
    
    def __init__(self, config=None):
        """Initialize intersection-specific cooperative policy."""
        super().__init__(config)
        
        # Intersection-specific social law parameters
        if config and 'social_laws' in config:
            social_laws = config['social_laws']
            
            # Polite Gap Provision parameters
            if 'polite_gap_provision' in social_laws:
                gap_config = social_laws['polite_gap_provision']
                self.gap_extension_time = gap_config.get('gap_extension_time', 1.5)
                self.gap_detection_range = gap_config.get('detection_range', 40.0)
                self.gap_speed_reduction = gap_config.get('speed_reduction_factor', 0.85)
                self.gap_provision_enabled = gap_config.get('enabled', True)
            else:
                self.gap_extension_time = 1.5
                self.gap_detection_range = 40.0
                self.gap_speed_reduction = 0.85
                self.gap_provision_enabled = True
                
            # Cooperative Turn-Taking parameters
            if 'cooperative_turn_taking' in social_laws:
                turn_config = social_laws['cooperative_turn_taking']
                self.max_consecutive_through = turn_config.get('max_consecutive_through', 3)
                self.turn_wait_threshold = turn_config.get('turn_wait_threshold', 5.0)
                self.courtesy_gap_size = turn_config.get('courtesy_gap_size', 8.0)
                self.turn_taking_enabled = turn_config.get('enabled', True)
            else:
                self.max_consecutive_through = 3
                self.turn_wait_threshold = 5.0
                self.courtesy_gap_size = 8.0
                self.turn_taking_enabled = True
                
            # Adaptive Right-of-Way parameters
            if 'adaptive_right_of_way' in social_laws:
                row_config = social_laws['adaptive_right_of_way']
                self.base_wait_time = row_config.get('base_wait_time', 3.0)
                self.wait_time_multiplier = row_config.get('wait_time_multiplier', 1.2)
                self.emergency_override = row_config.get('emergency_override', True)
                self.adaptive_row_enabled = row_config.get('enabled', True)
            else:
                self.base_wait_time = 3.0
                self.wait_time_multiplier = 1.2
                self.emergency_override = True
                self.adaptive_row_enabled = True
        else:
            # Default parameters
            self.gap_extension_time = 1.5
            self.gap_detection_range = 40.0
            self.gap_speed_reduction = 0.85
            self.gap_provision_enabled = True
            self.max_consecutive_through = 3
            self.turn_wait_threshold = 5.0
            self.courtesy_gap_size = 8.0
            self.turn_taking_enabled = True
            self.base_wait_time = 3.0
            self.wait_time_multiplier = 1.2
            self.emergency_override = True
            self.adaptive_row_enabled = True
        
        # Internal state for intersection-specific behaviors
        self.consecutive_through_count = 0
        self.last_turn_assistance_step = 0
        
        logging.info("IntersectionCooperativePolicy initialized with intersection-specific social laws")
    
    def act(self, obs):
        """
        Enhanced action selection with intersection-specific cooperative behaviors.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return super().act(obs)
        
        # Analyze intersection context
        intersection_context = self._analyze_intersection_context(obs)
        
        # Check for intersection-specific social law opportunities
        if self._should_provide_gap_for_turner(intersection_context):
            return self._execute_gap_provision_maneuver(obs)
        
        if self._should_facilitate_turn_taking(intersection_context):
            return self._execute_turn_taking_assistance(obs)
        
        if self._should_apply_adaptive_right_of_way(intersection_context):
            return self._execute_adaptive_right_of_way(obs)
        
        # Fall back to base cooperative policy
        return super().act(obs)
    
    def _analyze_intersection_context(self, obs):
        """
        Analyze the intersection traffic context to identify social law opportunities.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            dict: Context information about intersection traffic
        """
        context = {
            'turning_vehicles': [],
            'waiting_vehicles': [],
            'conflict_vehicles': [],
            'traffic_density': 0,
            'ego_position': None,
            'approaching_intersection': False
        }
        
        if len(obs) == 0:
            return context
        
        # Ego vehicle information
        ego_info = obs[0]
        context['ego_position'] = {
            'x': ego_info[1] if len(ego_info) > 1 else 0,
            'y': ego_info[2] if len(ego_info) > 2 else 0,
            'vx': ego_info[3] if len(ego_info) > 3 else 20,
            'vy': ego_info[4] if len(ego_info) > 4 else 0
        }
        
        # Detect if approaching intersection (simplified heuristic)
        # In a real intersection, this would use more sophisticated detection
        context['approaching_intersection'] = abs(context['ego_position']['x']) < 50
        
        # Analyze other vehicles
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                vehicle_info = {
                    'index': i,
                    'x': vehicle[1],
                    'y': vehicle[2], 
                    'vx': vehicle[3] if len(vehicle) > 3 else 20,
                    'vy': vehicle[4] if len(vehicle) > 4 else 0
                }
                
                # Detect turning vehicles (simplified - based on lateral velocity)
                if abs(vehicle_info['vy']) > 2.0:  # Significant lateral movement
                    context['turning_vehicles'].append(vehicle_info)
                
                # Detect waiting vehicles (low speed near intersection)
                if abs(vehicle_info['vx']) < 5.0 and abs(vehicle_info['x']) < 30:
                    context['waiting_vehicles'].append(vehicle_info)
                
                # Detect potential conflict vehicles
                if abs(vehicle_info['x']) < 40 and abs(vehicle_info['y']) < 20:
                    context['conflict_vehicles'].append(vehicle_info)
        
        context['traffic_density'] = len(context['conflict_vehicles'])
        
        return context
    
    def _should_provide_gap_for_turner(self, context):
        """
        Determine if we should provide a gap for a turning vehicle.
        
        Args:
            context: Intersection context from _analyze_intersection_context
            
        Returns:
            bool: True if gap provision is beneficial
        """
        if not self.gap_provision_enabled or not context['approaching_intersection']:
            return False
        
        # Check for turning vehicles that could benefit from a gap
        for turner in context['turning_vehicles']:
            distance = abs(turner['x'])
            if distance < self.gap_detection_range:
                # Check if we're in a position to help
                ego_x = context['ego_position']['x']
                if ego_x > turner['x'] and ego_x - turner['x'] < 20:  # We're ahead and close
                    return True
        
        return False
    
    def _should_facilitate_turn_taking(self, context):
        """
        Determine if we should facilitate turn-taking behavior.
        
        Args:
            context: Intersection context
            
        Returns:
            bool: True if turn-taking assistance is needed
        """
        if not self.turn_taking_enabled:
            return False
        
        # Check if we've had too many consecutive through movements
        if self.consecutive_through_count >= self.max_consecutive_through:
            # Look for waiting turning vehicles
            for waiter in context['waiting_vehicles']:
                if abs(waiter['x']) < 25:  # Close to intersection
                    return True
        
        return False
    
    def _should_apply_adaptive_right_of_way(self, context):
        """
        Determine if adaptive right-of-way should be applied.
        
        Args:
            context: Intersection context
            
        Returns:
            bool: True if adaptive right-of-way is beneficial
        """
        if not self.adaptive_row_enabled:
            return False
        
        # Check for vehicles that have been waiting longer than threshold
        # (In a real implementation, this would track actual wait times)
        long_waiting_vehicles = [v for v in context['waiting_vehicles'] 
                               if abs(v['vx']) < 2.0]  # Very slow/stopped vehicles
        
        return len(long_waiting_vehicles) > 0
    
    def _execute_gap_provision_maneuver(self, obs):
        """
        Execute gap provision by controlled deceleration.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action to create gap
        """
        # Controlled deceleration to create gap
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 20
        
        # If moving fast enough, slow down to create gap
        if ego_speed > 10:
            logging.debug("Executing gap provision maneuver for turning vehicle")
            return 4  # SLOWER action
        else:
            return 1  # IDLE - maintain gap
    
    def _execute_turn_taking_assistance(self, obs):
        """
        Execute turn-taking assistance by yielding right-of-way.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action to assist turn-taking
        """
        # Yield by slowing down and resetting consecutive count
        self.consecutive_through_count = 0
        logging.debug("Executing turn-taking assistance")
        return 4  # SLOWER to yield right-of-way
    
    def _execute_adaptive_right_of_way(self, obs):
        """
        Execute adaptive right-of-way by extending courtesy.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action for adaptive right-of-way
        """
        # Extended courtesy through speed reduction
        logging.debug("Executing adaptive right-of-way extension")
        return 4  # SLOWER to extend courtesy
    
    def _update_intersection_state(self, action):
        """
        Update internal state based on action taken.
        
        Args:
            action: Action taken this step
        """
        # Track consecutive through movements
        if action == 1 or action == 3:  # IDLE or FASTER (continuing through)
            self.consecutive_through_count += 1
        elif action == 4:  # SLOWER (yielding behavior)
            self.consecutive_through_count = max(0, self.consecutive_through_count - 1)


class IntersectionSelfishPolicy:
    """
    Selfish policy specialized for intersection scenarios.
    
    Extends basic selfish behavior with intersection-specific optimizations
    while maintaining selfish characteristics.
    """
    
    def __init__(self, config=None):
        """Initialize intersection-specific selfish policy."""
        from .selfish_policy import SelfishPolicy
        self.base_policy = SelfishPolicy(config)
        
        # Intersection-specific parameters (more aggressive than cooperative)
        self.intersection_aggressiveness = 1.2  # More aggressive at intersections
        self.right_of_way_assertion = True     # Assert right-of-way strongly
        self.gap_acceptance_threshold = 0.3    # Lower threshold (more aggressive)
        
        logging.info("IntersectionSelfishPolicy initialized with intersection-specific parameters")
    
    def act(self, obs):
        """
        Action selection with intersection-specific selfish optimizations.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        # Use base selfish policy but with intersection adjustments
        base_action = self.base_policy.act(obs)
        
        # At intersections, be more assertive about maintaining speed
        if self._is_at_intersection(obs):
            if base_action == 4:  # Don't slow down as much at intersections
                return 1  # IDLE instead of SLOWER
        
        return base_action
    
    def _is_at_intersection(self, obs):
        """
        Simple heuristic to detect intersection proximity.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            bool: True if near intersection
        """
        if len(obs) == 0:
            return False
        
        ego_info = obs[0]
        ego_x = ego_info[1] if len(ego_info) > 1 else 0
        
        # Simple heuristic: intersection if x position is near center
        return abs(ego_x) < 30