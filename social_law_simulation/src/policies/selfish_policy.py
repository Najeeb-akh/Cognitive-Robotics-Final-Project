"""
Selfish Agent Policy Implementation (FR2)

This module implements the baseline "selfish" agent that follows standard
traffic models without any cooperative behaviors. Agents maximize their
own velocity using IDM for car-following and MOBIL for lane changing.
"""

import numpy as np


class SelfishPolicy:
    """
    Selfish agent policy that implements standard IDM and MOBIL models.
    
    Key characteristics:
    - Maximizes forward velocity without cooperation
    - Uses standard IDM for longitudinal control
    - Uses MOBIL for lane changing (personal gain only)
    - Treats other vehicles as dynamic obstacles
    - Returns highway-env action indices
    """
    
    def __init__(self, config=None):
        """Initialize selfish policy with configuration parameters."""
        
        # IDM parameters from config or defaults
        if config and 'baseline' in config and 'idm' in config['baseline']:
            idm_config = config['baseline']['idm']
            self.TIME_HEADWAY = idm_config.get('time_headway', 1.5)
            self.MAX_ACCELERATION = idm_config.get('max_acceleration', 3.0)
            self.COMFORTABLE_DECELERATION = idm_config.get('comfortable_deceleration', 3.0)
            self.DESIRED_VELOCITY = idm_config.get('desired_velocity', 30.0)
            self.MINIMUM_SPACING = idm_config.get('minimum_spacing', 2.0)
        else:
            # Default IDM parameters
            self.TIME_HEADWAY = 1.5
            self.MAX_ACCELERATION = 3.0
            self.COMFORTABLE_DECELERATION = 3.0
            self.DESIRED_VELOCITY = 30.0
            self.MINIMUM_SPACING = 2.0
            
        # MOBIL parameters from config or defaults
        if config and 'baseline' in config and 'mobil' in config['baseline']:
            mobil_config = config['baseline']['mobil']
            self.POLITENESS_FACTOR = mobil_config.get('politeness_factor', 0.1)
            self.LANE_CHANGE_THRESHOLD = mobil_config.get('lane_change_threshold', 0.2)
            self.MAX_SAFE_DECELERATION = mobil_config.get('max_safe_deceleration', 4.0)
        else:
            # Default MOBIL parameters (low politeness for selfish behavior)
            self.POLITENESS_FACTOR = 0.1  # Low politeness
            self.LANE_CHANGE_THRESHOLD = 0.2
            self.MAX_SAFE_DECELERATION = 4.0
    
    def act(self, obs):
        """
        Main action function that implements selfish driving behavior.
        
        This method analyzes the environment observation and returns the optimal
        action according to selfish (IDM/MOBIL) decision-making.
        
        Args:
            obs: Highway-env observation array [vehicles_count, features]
            
        Returns:
            int: Action index for highway-env (0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER)
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
        
        # Prioritize speed control over lane changes for more stable behavior
        speed_action = self._determine_speed_action_from_obs(obs)
        
        # Only consider lane changes if current speed situation is acceptable
        if speed_action == 1:  # IDLE - speed is fine, consider lane change
            lane_change_direction = self._mobil_lane_change_from_obs(obs)
            
            # Convert to highway-env action format
            if lane_change_direction == -1:  # Change to left lane
                return 0  # LANE_LEFT
            elif lane_change_direction == 1:  # Change to right lane
                return 2  # LANE_RIGHT
        
        # Default to speed action (more stable)
        return speed_action
    
    def _determine_speed_action_from_obs(self, obs):
        """
        Determine speed action (FASTER/SLOWER/IDLE) based on observation.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action (1=IDLE, 3=FASTER, 4=SLOWER)
        """
        if len(obs) < 2:
            return 3  # FASTER if no other vehicles visible
            
        # Find front vehicle in same lane
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 20.0  # vx, default speed if not available
        
        front_vehicle = self._find_front_vehicle_from_obs(obs)
        
        if front_vehicle is None:
            # No vehicle ahead, accelerate to desired velocity
            if ego_speed < self.DESIRED_VELOCITY * 0.9:
                return 3  # FASTER
            else:
                return 1  # IDLE
        
        # Calculate distance and relative speed
        rel_x = front_vehicle[1]  # Relative x position 
        rel_vx = front_vehicle[3] - ego_info[3]  # Relative velocity
        
        # Convert relative position to actual distance
        distance = max(abs(rel_x), 2.0)  # Minimum safe distance
        
        # Simple following logic based on distance and speed difference
        safe_distance = self.MINIMUM_SPACING + self.TIME_HEADWAY * ego_speed
        
        if distance < safe_distance:
            return 4  # SLOWER - too close
        elif distance > safe_distance * 2 and ego_speed < self.DESIRED_VELOCITY * 0.9:
            return 3  # FASTER - safe to accelerate
        else:
            return 1  # IDLE - maintain current speed
    
    def _mobil_lane_change_from_obs(self, obs):
        """
        Determine if a lane change is beneficial using observation-based MOBIL model.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: -1 for left lane change, 1 for right, 0 for stay
        """
        if len(obs) < 2:
            return 0  # No other vehicles visible, stay in lane
            
        # Analyze current lane situation
        current_front = self._find_front_vehicle_from_obs(obs)
        current_utility = self._calculate_lane_utility(current_front, obs[0])
        
        # Check left lane change opportunity
        left_vehicles = self._find_vehicles_in_adjacent_lane(obs, -1)  # Left
        left_utility = self._calculate_adjacent_lane_utility(left_vehicles, obs[0])
        
        # Check right lane change opportunity  
        right_vehicles = self._find_vehicles_in_adjacent_lane(obs, 1)  # Right
        right_utility = self._calculate_adjacent_lane_utility(right_vehicles, obs[0])
        
        # Simple lane change decision based on utility difference
        left_benefit = left_utility - current_utility
        right_benefit = right_utility - current_utility
        
        # Make lane changes more conservative by increasing threshold and adding cooldown
        higher_threshold = self.LANE_CHANGE_THRESHOLD * 2.0  # More conservative threshold
        
        # Check if lane change is safe and beneficial
        if left_benefit > higher_threshold and self._is_lane_change_safe_from_obs(left_vehicles):
            return -1
        elif right_benefit > higher_threshold and self._is_lane_change_safe_from_obs(right_vehicles):
            return 1
        
        return 0  # Stay in current lane
    
    def _find_front_vehicle_from_obs(self, obs):
        """
        Find the closest vehicle ahead in the current lane from observation.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            array or None: Front vehicle observation or None if not found
        """
        ego_y = obs[0][2] if len(obs[0]) > 2 else 0  # Ego y position
        
        front_vehicles = []
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                rel_x, rel_y = vehicle[1], vehicle[2]
                # Check if vehicle is ahead (positive x) and in same lane (similar y)
                if rel_x > 0 and abs(rel_y) < 2.0:  # Same lane threshold
                    front_vehicles.append((rel_x, vehicle))
        
        if front_vehicles:
            # Return closest front vehicle
            return min(front_vehicles, key=lambda x: x[0])[1]
        
        return None
    
    def _find_vehicles_in_adjacent_lane(self, obs, direction):
        """
        Find vehicles in adjacent lane (left=-1, right=1).
        
        Args:
            obs: Highway-env observation array
            direction: -1 for left lane, 1 for right lane
            
        Returns:
            list: List of vehicles in adjacent lane
        """
        vehicles = []
        lane_width = 4.0  # Typical highway lane width
        target_y = direction * lane_width
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                rel_x, rel_y = vehicle[1], vehicle[2]
                # Check if vehicle is in target adjacent lane
                if abs(rel_y - target_y) < 2.0:  # Lane tolerance
                    vehicles.append(vehicle)
        
        return vehicles
    
    def _calculate_lane_utility(self, front_vehicle, ego_info):
        """
        Calculate utility of current lane based on front vehicle.
        
        Args:
            front_vehicle: Front vehicle observation or None
            ego_info: Ego vehicle information
            
        Returns:
            float: Lane utility (higher is better)
        """
        if front_vehicle is None:
            return 0.8  # Good but not perfect utility if no obstacles
        
        distance = abs(front_vehicle[1])  # Distance to front vehicle
        rel_speed = front_vehicle[3] - ego_info[3]  # Relative speed
        
        # More conservative utility calculation
        utility = (distance / 100.0) + (rel_speed / 20.0)  # Scaled down
        return max(min(utility, 1.0), 0.0)  # Clamp between 0 and 1
    
    def _calculate_adjacent_lane_utility(self, vehicles, ego_info):
        """
        Calculate utility of adjacent lane.
        
        Args:
            vehicles: List of vehicles in adjacent lane
            ego_info: Ego vehicle information
            
        Returns:
            float: Lane utility (higher is better)
        """
        if not vehicles:
            return 1.0  # High utility if lane is empty
        
        # Find closest vehicle ahead in adjacent lane
        front_vehicle = None
        min_distance = float('inf')
        
        for vehicle in vehicles:
            rel_x = vehicle[1]
            if rel_x > 0 and rel_x < min_distance:  # Vehicle ahead
                min_distance = rel_x
                front_vehicle = vehicle
        
        if front_vehicle is None:
            return 0.6  # Moderate utility if no vehicle ahead
        
        return self._calculate_lane_utility(front_vehicle, ego_info)
    
    def _is_lane_change_safe_from_obs(self, vehicles, ego_info=None):
        """
        Check if lane change is safe based on vehicles in target lane.
        
        Args:
            vehicles: List of vehicles in target lane
            ego_info: Ego vehicle information (optional, not used in selfish policy)
            
        Returns:
            bool: True if safe, False otherwise
        """
        min_safe_distance = 15.0  # Minimum safe distance in meters
        
        for vehicle in vehicles:
            distance = abs(vehicle[1])  # Distance to vehicle
            if distance < min_safe_distance:
                return False
        
        return True


# Compatibility alias for existing code
SelfishAgent = SelfishPolicy