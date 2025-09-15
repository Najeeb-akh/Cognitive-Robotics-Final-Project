"""
Parking Lot Policy Implementation

This module implements parking lot-specific policies that extend the baseline
behaviors with parking assistance, space optimization, and courteous maneuvering.
"""

import numpy as np
from .selfish_policy import SelfishPolicy
from .cooperative_policy import CooperativePolicy


class ParkingLotSelfishPolicy(SelfishPolicy):
    """
    Selfish policy optimized for parking lot scenarios.
    
    Extends the base selfish policy with parking-specific behaviors:
    - Aggressive space claiming
    - Competitive maneuvering
    - Minimal courtesy for other vehicles
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Parking-specific selfish parameters
        self.PARKING_AGGRESSIVENESS = 0.8  # Higher = more aggressive
        self.SPACE_CLAIMING_DISTANCE = 15.0  # Distance to claim parking spaces
        self.MANEUVERING_PATIENCE = 2.0  # Seconds to wait before forcing maneuvers
        
        # Load parking-specific config if available
        if config and 'environment' in config:
            parking_config = config['environment'].get('parking_lot', {})
            self.PARKING_AGGRESSIVENESS = parking_config.get('aggressiveness', 0.8)
            self.SPACE_CLAIMING_DISTANCE = parking_config.get('space_claiming_distance', 15.0)
    
    def act(self, obs):
        """
        Main action function for parking lot selfish behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
        
        # Apply parking-specific selfish behaviors
        parking_action = self._apply_parking_selfish_behavior(obs)
        if parking_action is not None:
            return parking_action
        
        # Fall back to base selfish behavior
        return super().act(obs)
    
    def _apply_parking_selfish_behavior(self, obs):
        """
        Apply parking-specific selfish behaviors.
        
        Args:
            obs: Observation array
            
        Returns:
            int or None: Action if parking behavior applies, None otherwise
        """
        # Check for parking space opportunities
        if self._detect_parking_opportunity(obs):
            return self._claim_parking_space(obs)
        
        # Check for competitive maneuvering
        if self._detect_maneuvering_competition(obs):
            return self._competitive_maneuver(obs)
        
        return None
    
    def _detect_parking_opportunity(self, obs):
        """Detect if there's a parking opportunity nearby."""
        # Simplified detection - in real implementation, would analyze obs more carefully
        return len(obs) > 1 and np.random.random() < 0.1  # 10% chance of detecting opportunity
    
    def _claim_parking_space(self, obs):
        """Aggressively claim a parking space."""
        # Aggressive lane change to claim space
        return 0  # LANE_LEFT for aggressive positioning
    
    def _detect_maneuvering_competition(self, obs):
        """Detect if another vehicle is competing for the same space."""
        return len(obs) > 2 and np.random.random() < 0.05  # 5% chance of competition
    
    def _competitive_maneuver(self, obs):
        """Perform competitive maneuvering to gain advantage."""
        # Accelerate to get ahead of competition
        return 3  # FASTER


class ParkingLotCooperativePolicy(CooperativePolicy):
    """
    Cooperative policy with parking-specific social laws.
    
    Extends the base cooperative policy with parking assistance behaviors:
    - Parking assistance
    - Space optimization
    - Courteous maneuvering
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Parking-specific cooperative parameters
        self.PARKING_ASSISTANCE_ENABLED = True
        self.SPACE_OPTIMIZATION_ENABLED = True
        self.COURTEOUS_MANEUVERING_ENABLED = True
        
        # Load parking-specific social law parameters
        if config and 'social_laws' in config:
            social_config = config['social_laws']
            
            # Parking assistance parameters
            parking_assist = social_config.get('parking_assistance', {})
            self.PARKING_ASSISTANCE_ENABLED = parking_assist.get('enabled', True)
            self.ASSISTANCE_DISTANCE = parking_assist.get('assistance_distance', 20.0)
            self.COURTESY_WAIT_TIME = parking_assist.get('courtesy_wait_time', 3.0)
            
            # Space optimization parameters
            space_opt = social_config.get('space_optimization', {})
            self.SPACE_OPTIMIZATION_ENABLED = space_opt.get('enabled', True)
            self.OPTIMAL_SPACING_FACTOR = space_opt.get('optimal_spacing_factor', 1.2)
            self.EFFICIENCY_THRESHOLD = space_opt.get('efficiency_threshold', 0.8)
    
    def act(self, obs):
        """
        Main action function for parking lot cooperative behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return 1  # IDLE if no observation
        
        # Apply parking-specific cooperative behaviors
        parking_action = self._apply_parking_cooperative_behavior(obs)
        if parking_action is not None:
            return parking_action
        
        # Fall back to base cooperative behavior
        return super().act(obs)
    
    def _apply_parking_cooperative_behavior(self, obs):
        """
        Apply parking-specific cooperative behaviors.
        
        Args:
            obs: Observation array
            
        Returns:
            int or None: Action if parking behavior applies, None otherwise
        """
        # Check for parking assistance opportunities
        if self.PARKING_ASSISTANCE_ENABLED and self._detect_parking_assistance_need(obs):
            return self._provide_parking_assistance(obs)
        
        # Check for space optimization opportunities
        if self.SPACE_OPTIMIZATION_ENABLED and self._detect_space_optimization_opportunity(obs):
            return self._optimize_space_usage(obs)
        
        # Check for courteous maneuvering opportunities
        if self.COURTEOUS_MANEUVERING_ENABLED and self._detect_maneuvering_courtesy_opportunity(obs):
            return self._provide_maneuvering_courtesy(obs)
        
        return None
    
    def _detect_parking_assistance_need(self, obs):
        """Detect if another vehicle needs parking assistance."""
        # Simplified detection - would analyze obs for vehicles struggling to park
        return len(obs) > 1 and np.random.random() < 0.15  # 15% chance of detecting need
    
    def _provide_parking_assistance(self, obs):
        """Provide assistance to vehicles trying to park."""
        # Slow down to create space for parking maneuver
        return 4  # SLOWER
    
    def _detect_space_optimization_opportunity(self, obs):
        """Detect opportunities to optimize space usage."""
        # Check if current positioning could be more efficient
        return len(obs) > 2 and np.random.random() < 0.1  # 10% chance of optimization opportunity
    
    def _optimize_space_usage(self,obs):
        """Optimize space usage for better overall efficiency."""
        # Adjust position to create more efficient spacing
        return 1  # IDLE to maintain optimal position
    
    def _detect_maneuvering_courtesy_opportunity(self, obs):
        """Detect opportunities to provide courteous maneuvering."""
        # Check if another vehicle is trying to maneuver
        return len(obs) > 1 and np.random.random() < 0.12  # 12% chance of courtesy opportunity
    
    def _provide_maneuvering_courtesy(self, obs):
        """Provide courteous maneuvering assistance."""
        # Yield right of way or create space for maneuvering vehicle
        return 4  # SLOWER to yield
