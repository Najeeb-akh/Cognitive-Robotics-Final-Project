"""
Defensive Agent Policy Implementation

This module implements a "defensive/paranoid" agent that prioritizes safety
above all else, creating extreme caution and large safety margins.
This behavior creates dramatic trade-offs between safety and efficiency.
"""

import numpy as np
from .selfish_policy import SelfishPolicy


class DefensivePolicy(SelfishPolicy):
    """
    Defensive agent policy that prioritizes safety above all else.
    
    Key characteristics:
    - Extremely large following distances (3-5x normal)
    - Very slow speeds (50-70% of speed limit)
    - Never takes risks or makes aggressive moves
    - Creates traffic jams but maintains zero accidents
    - Tests the limits of safety vs efficiency trade-offs
    
    This behavior is the opposite of aggressive/competitive behavior
    and creates fascinating dynamics when mixed with selfish/cooperative agents.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Defensive parameters - much more conservative than selfish
        self.DEFENSIVE_TIME_HEADWAY = 4.0  # 4x normal headway (vs 1.5)
        self.DEFENSIVE_MAX_SPEED = 0.6     # 60% of desired speed
        self.DEFENSIVE_FOLLOWING_DISTANCE = 5.0  # 5x minimum spacing
        self.DEFENSIVE_LANE_CHANGE_THRESHOLD = 0.8  # Very high threshold
        self.DEFENSIVE_SPEED_REDUCTION_FACTOR = 0.7  # 30% speed reduction
        self.DEFENSIVE_CAUTION_DISTANCE = 50.0  # Start slowing 50m before obstacles
        
        # Load defensive-specific config if available
        if config and 'environment' in config:
            defensive_config = config['environment'].get('defensive', {})
            self.DEFENSIVE_TIME_HEADWAY = defensive_config.get('time_headway', 4.0)
            self.DEFENSIVE_MAX_SPEED = defensive_config.get('max_speed_factor', 0.6)
            self.DEFENSIVE_FOLLOWING_DISTANCE = defensive_config.get('following_distance', 5.0)
            self.DEFENSIVE_CAUTION_DISTANCE = defensive_config.get('caution_distance', 50.0)
        
        # Override base parameters with defensive values
        self.TIME_HEADWAY = self.DEFENSIVE_TIME_HEADWAY
        self.DESIRED_VELOCITY = self.DESIRED_VELOCITY * self.DEFENSIVE_MAX_SPEED
        self.MINIMUM_SPACING = self.MINIMUM_SPACING * self.DEFENSIVE_FOLLOWING_DISTANCE
        
        # Defensive state tracking
        self.defensive_mode_active = True
        self.last_safe_action = 1  # IDLE
        self.consecutive_cautious_actions = 0
        
    def act(self, obs):
        """
        Main action function that implements defensive driving behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
        
        # Apply defensive behavior analysis
        defensive_action = self._apply_defensive_behavior(obs)
        if defensive_action is not None:
            self.last_safe_action = defensive_action
            self.consecutive_cautious_actions += 1
            return defensive_action
        
        # If no specific defensive action needed, use very conservative base behavior
        conservative_action = self._conservative_base_behavior(obs)
        self.last_safe_action = conservative_action
        return conservative_action
    
    def _apply_defensive_behavior(self, obs):
        """
        Apply defensive driving behaviors based on observation.
        
        Args:
            obs: Observation array
            
        Returns:
            int or None: Defensive action if applicable, None otherwise
        """
        if len(obs) < 2:
            return None
            
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 0.0
        
        # Check for nearby vehicles that require defensive response
        nearby_threats = self._detect_nearby_threats(obs)
        
        if nearby_threats['front_vehicle_close']:
            # Slow down significantly when following too closely
            return 4  # SLOWER
            
        if nearby_threats['lateral_vehicles_close']:
            # Avoid lane changes when vehicles are nearby
            return 1  # IDLE
            
        if nearby_threats['high_speed_approaching']:
            # Slow down when high-speed vehicles are approaching
            return 4  # SLOWER
            
        if nearby_threats['complex_situation']:
            # In complex situations, always choose the safest option
            return 1  # IDLE
            
        return None
    
    def _detect_nearby_threats(self, obs):
        """
        Detect potential threats that require defensive response.
        
        Args:
            obs: Observation array
            
        Returns:
            dict: Threat assessment
        """
        threats = {
            'front_vehicle_close': False,
            'lateral_vehicles_close': False,
            'high_speed_approaching': False,
            'complex_situation': False
        }
        
        if len(obs) < 2:
            return threats
            
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 0.0
        
        # Check each observed vehicle
        for i, vehicle_info in enumerate(obs[1:], 1):
            if len(vehicle_info) < 5:
                continue
                
            presence, x, y, vx, vy = vehicle_info[:5]
            if presence < 0.5:  # Vehicle not present
                continue
                
            distance = np.sqrt(x**2 + y**2)
            relative_speed = np.sqrt(vx**2 + vy**2)
            
            # Front vehicle too close
            if abs(y) < 5.0 and x > 0 and distance < self.DEFENSIVE_CAUTION_DISTANCE:
                threats['front_vehicle_close'] = True
                
            # Lateral vehicles too close
            if abs(x) < 8.0 and abs(y) < 20.0:
                threats['lateral_vehicles_close'] = True
                
            # High-speed vehicle approaching
            if relative_speed > ego_speed * 1.5 and distance < 30.0:
                threats['high_speed_approaching'] = True
                
            # Complex situation (multiple vehicles nearby)
            if distance < 25.0:
                threats['complex_situation'] = True
                
        return threats
    
    def _conservative_base_behavior(self, obs):
        """
        Apply very conservative base driving behavior.
        
        Args:
            obs: Observation array
            
        Returns:
            int: Conservative action
        """
        if len(obs) < 2:
            return 1  # IDLE
            
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 0.0
        
        # Always prefer slower speeds
        if ego_speed > self.DESIRED_VELOCITY * 0.8:
            return 4  # SLOWER
            
        # Very rarely attempt lane changes, and only when absolutely safe
        if (self.consecutive_cautious_actions > 20 and 
            np.random.random() < 0.05):  # 5% chance after 20 cautious actions
            safe_lane_change = self._ultra_safe_lane_change(obs)
            if safe_lane_change is not None:
                self.consecutive_cautious_actions = 0
                return safe_lane_change
                
        # Default to maintaining current state
        return 1  # IDLE
    
    def _ultra_safe_lane_change(self, obs):
        """
        Perform ultra-safe lane change only when conditions are perfect.
        
        Args:
            obs: Observation array
            
        Returns:
            int or None: Lane change action if ultra-safe, None otherwise
        """
        if len(obs) < 3:
            return None
            
        # Check for completely clear lanes
        left_clear = self._is_lane_completely_clear(obs, 'left')
        right_clear = self._is_lane_completely_clear(obs, 'right')
        
        # Only change lanes if target lane is completely clear for 50+ meters
        if left_clear and np.random.random() < 0.3:  # 30% chance even if clear
            return 0  # LANE_LEFT
        elif right_clear and np.random.random() < 0.3:  # 30% chance even if clear
            return 2  # LANE_RIGHT
            
        return None
    
    def _is_lane_completely_clear(self, obs, direction):
        """
        Check if a lane is completely clear for defensive driving.
        
        Args:
            obs: Observation array
            direction: 'left' or 'right'
            
        Returns:
            bool: True if lane is completely clear
        """
        if len(obs) < 2:
            return False
            
        lane_threshold = 3.5 if direction == 'left' else -3.5
        
        for vehicle_info in obs[1:]:
            if len(vehicle_info) < 5:
                continue
                
            presence, x, y, vx, vy = vehicle_info[:5]
            if presence < 0.5:
                continue
                
            # Check if vehicle is in target lane and within 50 meters
            if ((direction == 'left' and x > lane_threshold) or 
                (direction == 'right' and x < lane_threshold)):
                distance = np.sqrt(x**2 + y**2)
                if distance < 50.0:  # 50 meter safety buffer
                    return False
                    
        return True
    
    def get_behavior_description(self):
        """Get a description of this defensive behavior."""
        return {
            'type': 'defensive',
            'characteristics': [
                'Extremely large following distances',
                'Very slow speeds (60% of limit)',
                'Never takes risks',
                'Creates traffic jams but zero accidents',
                'Tests safety vs efficiency trade-offs'
            ],
            'parameters': {
                'time_headway': self.DEFENSIVE_TIME_HEADWAY,
                'max_speed_factor': self.DEFENSIVE_MAX_SPEED,
                'following_distance_multiplier': self.DEFENSIVE_FOLLOWING_DISTANCE,
                'caution_distance': self.DEFENSIVE_CAUTION_DISTANCE
            }
        }
