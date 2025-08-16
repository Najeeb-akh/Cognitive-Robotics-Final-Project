"""
Cooperative Agent Policy Implementation (FR3)

This module implements the cooperative agent that extends the baseline selfish
behavior with three social laws: Cooperative Merging, Polite Yielding, and
Phantom Jam Mitigation. All decisions are made based on local observations only.
"""

import numpy as np
from .selfish_policy import SelfishPolicy


class CooperativePolicy(SelfishPolicy):
    """
    Cooperative agent policy that extends SelfishPolicy with social laws.
    
    Implements three social laws:
    - FR3.1: Cooperative Merging
    - FR3.2: Polite Yielding for Lane Changes  
    - FR3.3: Phantom Jam Mitigation
    
    Maintains decentralized approach - all decisions based on local observations only.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Social law parameters from config
        if config and 'social_laws' in config:
            social_config = config['social_laws']
            
            # Cooperative merging parameters
            coop_merge = social_config.get('cooperative_merging', {})
            self.COOP_MERGE_ENABLED = coop_merge.get('enabled', True)
            self.MERGE_DECEL_FACTOR = coop_merge.get('deceleration_factor', 0.8)
            self.MERGE_DETECTION_DISTANCE = coop_merge.get('detection_distance', 30.0)
            
            # Polite yielding parameters
            polite_yield = social_config.get('polite_yielding', {})
            self.POLITE_YIELD_ENABLED = polite_yield.get('enabled', True)
            self.YIELD_SPEED_FACTOR = polite_yield.get('speed_reduction_factor', 0.9)
            self.GAP_CREATION_TIME = polite_yield.get('gap_creation_time', 2.0)
            
            # Phantom jam mitigation parameters
            phantom_jam = social_config.get('phantom_jam_mitigation', {})
            self.PHANTOM_JAM_ENABLED = phantom_jam.get('enabled', True)
            self.DENSITY_THRESHOLD = phantom_jam.get('density_threshold', 40)
            self.INCREASED_TIME_HEADWAY = phantom_jam.get('increased_time_headway', 2.0)
            self.DEFAULT_TIME_HEADWAY = phantom_jam.get('default_time_headway', 1.5)
        else:
            # Default social law parameters
            self.COOP_MERGE_ENABLED = True
            self.MERGE_DECEL_FACTOR = 0.8
            self.MERGE_DETECTION_DISTANCE = 30.0
            
            self.POLITE_YIELD_ENABLED = True
            self.YIELD_SPEED_FACTOR = 0.9
            self.GAP_CREATION_TIME = 2.0
            
            self.PHANTOM_JAM_ENABLED = True
            self.DENSITY_THRESHOLD = 40
            self.INCREASED_TIME_HEADWAY = 2.0
            self.DEFAULT_TIME_HEADWAY = 1.5
            
        # State for tracking cooperative behaviors
        self.yielding_timer = 0.0
        self.cooperative_behavior_active = False
    
    def act(self, obs):
        """
        Main action function that implements cooperative behaviors.
        
        First checks for triggers of social laws, then falls back to
        selfish behavior if no cooperative action is needed.
        
        Args:
            obs: Highway-env observation array [vehicles_count, features]
            
        Returns:
            int: Action index for highway-env (0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER)
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
            
        # Update timer for yielding behavior
        if self.yielding_timer > 0:
            self.yielding_timer -= 1/30.0  # Assuming 30 FPS
            
        # Check social law triggers in order of priority
        cooperative_action = None
        
        # FR3.1: Cooperative Merging
        if self.COOP_MERGE_ENABLED:
            cooperative_action = self._check_cooperative_merging_from_obs(obs)
            if cooperative_action is not None:
                self.cooperative_behavior_active = True
                return cooperative_action
        
        # FR3.2: Polite Yielding for Lane Changes
        if self.POLITE_YIELD_ENABLED:
            cooperative_action = self._check_polite_yielding_from_obs(obs)
            if cooperative_action is not None:
                self.cooperative_behavior_active = True
                return cooperative_action
        
        # FR3.3: Phantom Jam Mitigation (modifies IDM parameters)
        if self.PHANTOM_JAM_ENABLED:
            self._check_phantom_jam_mitigation_from_obs(obs)
        
        # No cooperative behavior needed, use selfish policy
        self.cooperative_behavior_active = False
        return super().act(obs)
    
    def _check_cooperative_merging_from_obs(self, obs):
        """
        FR3.1: Cooperative Merging
        
        Trigger: Detect another agent in merging lane attempting to enter ahead
        Action: Moderately decelerate to create gap for safe merge
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int or None: Action to take, or None if no cooperation needed
        """
        # Look for vehicles in right lane (typical merge scenario)
        right_lane_vehicles = self._find_vehicles_in_adjacent_lane(obs, 1)  # Right lane
        
        for vehicle in right_lane_vehicles:
            rel_x, rel_y = vehicle[1], vehicle[2]
            
            # Check if vehicle is attempting to merge (slightly ahead and moving toward our lane)
            if 0 < rel_x < self.MERGE_DETECTION_DISTANCE:
                # Check if merge space is insufficient
                front_vehicle = self._find_front_vehicle_from_obs(obs)
                if self._is_merge_space_insufficient_from_obs(vehicle, front_vehicle, obs[0]):
                    return 4  # SLOWER action to create space
        
        return None
    
    
    def _check_polite_yielding_from_obs(self, obs):
        """
        FR3.2: Polite Yielding for Lane Changes
        
        Trigger: Detect adjacent agent signaling lane change to move in front
        Action: Slightly reduce speed to create safe gap
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int or None: Action to take, or None if no cooperation needed
        """
        # Check for vehicles in adjacent lanes wanting to change to our lane
        # Check left adjacent lane
        left_vehicles = self._find_vehicles_in_adjacent_lane(obs, -1)
        if self._detect_lane_change_intent_from_obs(left_vehicles, obs[0]):
            self.yielding_timer = self.GAP_CREATION_TIME
            return 4  # SLOWER action
        
        # Check right adjacent lane  
        right_vehicles = self._find_vehicles_in_adjacent_lane(obs, 1)
        if self._detect_lane_change_intent_from_obs(right_vehicles, obs[0]):
            self.yielding_timer = self.GAP_CREATION_TIME
            return 4  # SLOWER action
        
        # Continue yielding if timer is active
        if self.yielding_timer > 0:
            return 4  # SLOWER action
        
        return None
    
    def _detect_lane_change_intent_from_obs(self, adjacent_vehicles, ego_info):
        """
        Detect if a vehicle in adjacent lane wants to change to our lane.
        
        Args:
            adjacent_vehicles: List of vehicles in adjacent lane
            ego_info: Ego vehicle information
            
        Returns:
            bool: True if lane change intent detected
        """
        ego_speed = ego_info[3] if len(ego_info) > 3 else 0
        
        # Look for vehicles alongside or slightly behind/ahead
        for vehicle in adjacent_vehicles:
            rel_x = vehicle[1]  # Relative x position
            vehicle_speed = vehicle[3] if len(vehicle) > 3 else 0
            
            # Vehicle is alongside or close enough to want to change lanes
            if abs(rel_x) < 20.0:  # Within reasonable lane change distance
                # Simple heuristic: if they're slower, they likely want to change to faster lane
                if ego_speed > vehicle_speed * 1.1:  # Our lane is notably faster
                    return True
        
        return False
    
    def _check_phantom_jam_mitigation_from_obs(self, obs):
        """
        FR3.3: Phantom Jam Mitigation
        
        Trigger: Traffic density exceeds threshold
        Action: Increase time headway to absorb speed variations
        
        Args:
            obs: Highway-env observation array
        """
        # Calculate local traffic density from observation
        density = self._calculate_local_density_from_obs(obs)
        
        # Adjust time headway based on density
        if density > self.DENSITY_THRESHOLD:
            self.TIME_HEADWAY = self.INCREASED_TIME_HEADWAY
        else:
            self.TIME_HEADWAY = self.DEFAULT_TIME_HEADWAY
    
    def _check_phantom_jam_mitigation(self):
        """
        FR3.3: Phantom Jam Mitigation
        
        Trigger: Traffic density exceeds threshold
        Action: Increase time headway to absorb speed variations
        
        This modifies the IDM time headway parameter used in acceleration calculation.
        """
        # Calculate local traffic density
        density = self._calculate_local_density()
        
        # Adjust time headway based on density
        if density > self.DENSITY_THRESHOLD:
            self.TIME_HEADWAY = self.INCREASED_TIME_HEADWAY
        else:
            self.TIME_HEADWAY = self.DEFAULT_TIME_HEADWAY
    
    def _calculate_local_density_from_obs(self, obs):
        """
        Calculate local traffic density around the vehicle from observation.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            float: Estimated density in vehicles/km/lane
        """
        # Count vehicles in same lane within reasonable range
        local_range = 100.0  # meters
        vehicles_in_range = 0
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                rel_x, rel_y = vehicle[1], vehicle[2]
                # Check if vehicle is in same lane and within range
                if abs(rel_y) < 2.0 and abs(rel_x) < local_range:  # Same lane
                    vehicles_in_range += 1
        
        # Calculate density as vehicles per km
        area_km = (2 * local_range) / 1000.0  # Convert to km
        density = vehicles_in_range / area_km if area_km > 0 else 0
        
        return density
    
    def _is_lane_change_safe_from_obs(self, vehicles, ego_info=None):
        """
        Check if lane change is safe based on vehicles in target lane.
        Uses time-based safety calculations instead of distance-only.
        Overrides the base class method with improved safety logic.
        
        Args:
            vehicles: List of vehicles in target lane
            ego_info: Ego vehicle information (optional, uses last observed ego state if None)
            
        Returns:
            bool: True if safe, False otherwise
        """
        min_safe_time = 3.0  # Minimum safe time-to-collision in seconds
        min_safe_distance = 10.0  # Fallback minimum distance in meters
        
        # Get ego vehicle speed for relative calculations
        ego_speed = 20.0  # Default fallback speed
        if ego_info is not None and len(ego_info) > 3:
            ego_speed = abs(ego_info[3])
        
        for vehicle in vehicles:
            distance = abs(vehicle[1])  # Distance to vehicle
            vehicle_speed = abs(vehicle[3]) if len(vehicle) > 3 else 20.0
            
            # Check minimum distance first
            if distance < min_safe_distance:
                return False
            
            # Calculate time-to-collision if vehicles are approaching
            relative_speed = abs(vehicle_speed - ego_speed)
            if relative_speed > 1.0:  # Only if significant speed difference
                time_to_collision = distance / relative_speed
                if time_to_collision < min_safe_time:
                    return False
        
        return True
    
    def _is_merge_space_insufficient_from_obs(self, merging_vehicle, front_vehicle, ego_info):
        """
        Determine if there's insufficient space for the merging vehicle.
        Uses improved calculations considering relative velocities.
        Overrides the base class method with enhanced merge logic.
        
        Args:
            merging_vehicle: Merging vehicle observation
            front_vehicle: Front vehicle observation or None
            ego_info: Ego vehicle information
            
        Returns:
            bool: True if space is insufficient for safe merge
        """
        if front_vehicle is None:
            return False  # Plenty of space if no vehicle ahead
            
        # Calculate available space between ego and front vehicle
        available_space = abs(front_vehicle[1])  # Distance to front vehicle
        
        # Get vehicle speeds with fallbacks
        merge_speed = abs(merging_vehicle[3]) if len(merging_vehicle) > 3 else 20.0
        ego_speed = abs(ego_info[3]) if len(ego_info) > 3 else 20.0
        front_speed = abs(front_vehicle[3]) if len(front_vehicle) > 3 else 20.0
        
        # Calculate time-based safety requirements
        merge_time_required = 3.0  # Seconds needed for safe merge
        
        # Calculate required space considering merge dynamics
        base_spacing = self.MINIMUM_SPACING * 2  # Safety buffers
        merge_distance = merge_speed * merge_time_required  # Distance merging vehicle travels
        
        # Add buffer for relative velocity differences
        velocity_buffer = abs(merge_speed - ego_speed) * self.TIME_HEADWAY
        
        safe_merge_space = base_spacing + merge_distance + velocity_buffer
        
        return available_space < safe_merge_space
    
    def _mobil_lane_change_from_obs(self, obs):
        """
        Determine if a lane change is beneficial using observation-based MOBIL model.
        Overrides base class method to use improved safety checks with ego_info.
        
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
        
        # Make lane changes more conservative by increasing threshold
        higher_threshold = self.LANE_CHANGE_THRESHOLD * 2.0  # More conservative threshold
        
        # Check if lane change is safe and beneficial (pass ego_info to improved safety check)
        if left_benefit > higher_threshold and self._is_lane_change_safe_from_obs(left_vehicles, obs[0]):
            return -1
        elif right_benefit > higher_threshold and self._is_lane_change_safe_from_obs(right_vehicles, obs[0]):
            return 1
        
        return 0  # Stay in current lane


# Compatibility alias for existing code
CooperativeAgent = CooperativePolicy