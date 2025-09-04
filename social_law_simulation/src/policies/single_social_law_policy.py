"""
Single Social Law Policy Implementation

This module implements a policy that applies only one specific social law to 
the base selfish behavior, allowing for precise comparison and analysis of 
individual social laws.
"""

import numpy as np
from .selfish_policy import SelfishPolicy


class SingleSocialLawPolicy(SelfishPolicy):
    """
    Policy that applies only one specific social law to selfish base behavior.
    
    This policy enables precise testing and comparison of individual social laws
    by isolating their effects on traffic flow and safety.
    """
    
    def __init__(self, social_law_name: str, config=None):
        """
        Initialize policy with specific social law.
        
        Args:
            social_law_name: Name of the social law to apply
            config: Configuration dictionary
        """
        super().__init__(config)
        self.social_law_name = social_law_name
        
        # Validate social law exists in config
        available_laws = []
        if config and 'social_laws' in config:
            available_laws = list(config['social_laws'].keys())
            if social_law_name not in available_laws:
                raise ValueError(f"Unknown social law '{social_law_name}'. Available: {available_laws}")
        
        # Initialize social law configuration
        self._configure_single_law(config)
        
    def _configure_single_law(self, config):
        """Configure only the specified social law, disable all others."""
        
        # List of all available social laws
        all_laws = [
            'cooperative_merging', 'polite_yielding', 'phantom_jam_mitigation',
            'polite_gap_provision', 'cooperative_turn_taking', 'adaptive_right_of_way',
            'entry_facilitation', 'smooth_flow_maintenance', 'exit_courtesy',
            'safe_overtaking_protocol', 'defensive_positioning', 'slipstream_cooperation'
        ]
        
        # Disable all laws first
        for law in all_laws:
            setattr(self, f"{law.upper()}_ENABLED", False)
        
        # Enable only the specified law and load its parameters
        if config and 'social_laws' in config:
            social_config = config['social_laws']
            law_config = social_config.get(self.social_law_name, {})
            
            # Set the enabled flag
            setattr(self, f"{self.social_law_name.upper()}_ENABLED", law_config.get('enabled', True))
            
            # Load parameters based on the specific social law
            self._load_social_law_parameters(law_config)
    
    def _load_social_law_parameters(self, law_config):
        """Load parameters for the specific social law."""
        
        if self.social_law_name == 'cooperative_merging':
            self.MERGE_DECEL_FACTOR = law_config.get('deceleration_factor', 0.8)
            self.MERGE_DETECTION_DISTANCE = law_config.get('detection_distance', 30.0)
            
        elif self.social_law_name == 'polite_yielding':
            self.YIELD_SPEED_FACTOR = law_config.get('speed_reduction_factor', 0.9)
            self.GAP_CREATION_TIME = law_config.get('gap_creation_time', 2.0)
            
        elif self.social_law_name == 'phantom_jam_mitigation':
            self.DENSITY_THRESHOLD = law_config.get('density_threshold', 40)
            self.INCREASED_TIME_HEADWAY = law_config.get('increased_time_headway', 2.0)
            self.DEFAULT_TIME_HEADWAY = law_config.get('default_time_headway', 1.5)
            
        elif self.social_law_name == 'polite_gap_provision':
            self.GAP_EXTENSION_TIME = law_config.get('gap_extension_time', 1.5)
            self.GAP_DETECTION_RANGE = law_config.get('detection_range', 40.0)
            self.GAP_SPEED_REDUCTION_FACTOR = law_config.get('speed_reduction_factor', 0.85)
            
        elif self.social_law_name == 'cooperative_turn_taking':
            self.MAX_CONSECUTIVE_THROUGH = law_config.get('max_consecutive_through', 3)
            self.TURN_WAIT_THRESHOLD = law_config.get('turn_wait_threshold', 5.0)
            self.COURTESY_GAP_SIZE = law_config.get('courtesy_gap_size', 8.0)
            
        elif self.social_law_name == 'adaptive_right_of_way':
            self.BASE_WAIT_TIME = law_config.get('base_wait_time', 3.0)
            self.WAIT_TIME_MULTIPLIER = law_config.get('wait_time_multiplier', 1.2)
            self.EMERGENCY_OVERRIDE = law_config.get('emergency_override', True)
            
        elif self.social_law_name == 'entry_facilitation':
            self.ENTRY_DETECTION_DISTANCE = law_config.get('entry_detection_distance', 50.0)
            self.FACILITATION_SPEED_FACTOR = law_config.get('facilitation_speed_factor', 0.9)
            self.GAP_CREATION_DISTANCE = law_config.get('gap_creation_distance', 15.0)
            
        elif self.social_law_name == 'smooth_flow_maintenance':
            self.TARGET_SPACING_FACTOR = law_config.get('target_spacing_factor', 1.8)
            self.SPEED_HARMONIZATION_RATE = law_config.get('speed_harmonization_rate', 0.1)
            self.ACCORDION_PREVENTION_THRESHOLD = law_config.get('accordion_prevention_threshold', 5.0)
            
        elif self.social_law_name == 'exit_courtesy':
            self.EARLY_SIGNAL_DISTANCE = law_config.get('early_signal_distance', 30.0)
            self.EXIT_GAP_PROVISION = law_config.get('exit_gap_provision', True)
            self.LANE_CHANGE_ASSISTANCE = law_config.get('lane_change_assistance', True)
            
        elif self.social_law_name == 'safe_overtaking_protocol':
            self.MIN_SPEED_DIFFERENTIAL = law_config.get('minimum_speed_differential', 10)
            self.SAFE_OVERTAKING_DISTANCE = law_config.get('safe_overtaking_distance', 50)
            self.ABORT_THRESHOLD = law_config.get('abort_threshold', 20)
            
        elif self.social_law_name == 'defensive_positioning':
            self.ALLOW_FASTER_PASS = law_config.get('allow_faster_pass', True)
            self.SPEED_DIFFERENTIAL_THRESHOLD = law_config.get('speed_differential_threshold', 15)
            self.DEFENSIVE_GAP_SIZE = law_config.get('defensive_gap_size', 20)
            
        elif self.social_law_name == 'slipstream_cooperation':
            self.COOPERATION_SPEED_RANGE = law_config.get('cooperation_speed_range', [80, 120])
            self.LEAD_CONSISTENCY_FACTOR = law_config.get('lead_consistency_factor', 0.95)
            self.ALTERNATING_LEAD_DISTANCE = law_config.get('alternating_lead_distance', 500)
    
    def act(self, obs):
        """
        Main action function that applies the single social law.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
        
        # Apply the specific social law if enabled
        social_action = self._apply_social_law(obs)
        if social_action is not None:
            return social_action
        
        # Fall back to base selfish behavior
        return super().act(obs)
    
    def _apply_social_law(self, obs):
        """Apply the specific social law and return action if applicable."""
        
        if self.social_law_name == 'cooperative_merging' and getattr(self, 'COOPERATIVE_MERGING_ENABLED', False):
            return self._apply_cooperative_merging(obs)
            
        elif self.social_law_name == 'polite_yielding' and getattr(self, 'POLITE_YIELDING_ENABLED', False):
            return self._apply_polite_yielding(obs)
            
        elif self.social_law_name == 'phantom_jam_mitigation' and getattr(self, 'PHANTOM_JAM_MITIGATION_ENABLED', False):
            return self._apply_phantom_jam_mitigation(obs)
            
        elif self.social_law_name == 'polite_gap_provision' and getattr(self, 'POLITE_GAP_PROVISION_ENABLED', False):
            return self._apply_polite_gap_provision(obs)
        
        # Add other social law implementations as needed
        return None
    
    def _apply_cooperative_merging(self, obs):
        """Apply cooperative merging behavior."""
        # Implementation based on CooperativePolicy
        ego_info = obs[0]
        ego_speed = ego_info[3] if len(ego_info) > 3 else 20.0
        
        # Look for vehicles that might be merging (vehicles in adjacent lanes with relative motion)
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                # Check if vehicle appears to be merging (lateral movement)
                relative_y = vehicle[2] - ego_info[2] if len(ego_info) > 2 else 0
                relative_x = vehicle[1] - ego_info[1] if len(ego_info) > 1 else 0
                
                # If vehicle is nearby laterally and ahead/behind within detection distance
                if abs(relative_y) < 6.0 and abs(relative_x) < self.MERGE_DETECTION_DISTANCE:
                    # Cooperative response: slow down to create space
                    return 4  # SLOWER
        
        return None  # No cooperative action needed
    
    def _apply_polite_yielding(self, obs):
        """Apply polite yielding behavior."""
        # Look for vehicles that might need to change lanes
        ego_info = obs[0]
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                relative_x = vehicle[1] - ego_info[1] if len(ego_info) > 1 else 0
                relative_y = vehicle[2] - ego_info[2] if len(ego_info) > 2 else 0
                
                # If vehicle is close behind in adjacent lane
                if abs(relative_y) > 2.0 and abs(relative_y) < 6.0 and relative_x < 0 and abs(relative_x) < 20.0:
                    # Polite response: reduce speed to create gap
                    return 4  # SLOWER
        
        return None
    
    def _apply_phantom_jam_mitigation(self, obs):
        """Apply phantom jam mitigation behavior."""
        # Count nearby vehicles to estimate density
        vehicle_count = sum(1 for vehicle in obs[1:] if vehicle[0] == 1)
        
        # If density is high, increase following distance
        if vehicle_count > self.DENSITY_THRESHOLD / 10:  # Scale for observation window
            front_vehicle = self._find_front_vehicle_from_obs(obs)
            if front_vehicle is not None:
                ego_info = obs[0]
                distance = abs(front_vehicle[1] - ego_info[1])
                ego_speed = ego_info[3] if len(ego_info) > 3 else 20.0
                
                # Use increased headway in dense conditions
                safe_distance = self.MINIMUM_SPACING + self.INCREASED_TIME_HEADWAY * ego_speed
                
                if distance < safe_distance:
                    return 4  # SLOWER
        
        return None
    
    def _apply_polite_gap_provision(self, obs):
        """Apply polite gap provision behavior."""
        ego_info = obs[0]
        
        # Look for vehicles that might need gaps
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                relative_x = vehicle[1] - ego_info[1] if len(ego_info) > 1 else 0
                relative_y = vehicle[2] - ego_info[2] if len(ego_info) > 2 else 0
                
                # If vehicle is in detection range and might need a gap
                if abs(relative_x) < self.GAP_DETECTION_RANGE and abs(relative_y) > 2.0:
                    # Provide gap by reducing speed
                    return 4  # SLOWER
        
        return None


# Scenario-specific single social law policies

class SingleSocialLawIntersectionPolicy(SingleSocialLawPolicy):
    """Single social law policy specialized for intersections."""
    
    def __init__(self, social_law_name: str, config=None):
        super().__init__(social_law_name, config)
        # Add intersection-specific initialization if needed


class SingleSocialLawRoundaboutPolicy(SingleSocialLawPolicy):
    """Single social law policy specialized for roundabouts."""
    
    def __init__(self, social_law_name: str, config=None):
        super().__init__(social_law_name, config)
        # Add roundabout-specific initialization if needed


class SingleSocialLawRacetrackPolicy(SingleSocialLawPolicy):
    """Single social law policy specialized for racetracks."""
    
    def __init__(self, social_law_name: str, config=None):
        super().__init__(social_law_name, config)
        # Add racetrack-specific initialization if needed