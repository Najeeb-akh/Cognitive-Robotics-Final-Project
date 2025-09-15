"""
Parking Lot Policy Implementation

This module implements parking lot-specific policies that extend the baseline
behaviors with parking assistance, space optimization, and courteous maneuvering.
"""

import numpy as np
import math
from .selfish_policy import SelfishPolicy
from .cooperative_policy import CooperativePolicy


class ParkingLotSelfishPolicy(SelfishPolicy):
    """
    Selfish policy optimized for parking lot scenarios.
    
    Extends the base selfish policy with parking-specific behaviors:
    - Targeted parking space navigation
    - Wall collision avoidance
    - Aggressive space claiming
    - Competitive maneuvering
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Parking-specific selfish parameters
        self.PARKING_AGGRESSIVENESS = 0.8  # Higher = more aggressive
        self.SPACE_CLAIMING_DISTANCE = 15.0  # Distance to claim parking spaces
        self.MANEUVERING_PATIENCE = 2.0  # Seconds to wait before forcing maneuvers
        
        # Navigation and collision avoidance parameters
        self.TARGET_PARKING_SPACE = None  # Will be set based on vehicle ID or observation
        self.WALL_DETECTION_DISTANCE = 8.0  # Distance to detect walls
        self.SAFE_DISTANCE_FROM_WALL = 3.0  # Minimum safe distance from walls
        self.PARKING_SPACE_REWARD = 5.0  # Reward for reaching target parking space
        self.WALL_COLLISION_PENALTY = -10.0  # Penalty for hitting walls
        
        # Load parking-specific config if available
        if config and 'environment' in config:
            parking_config = config['environment'].get('parking_lot', {})
            self.PARKING_AGGRESSIVENESS = parking_config.get('aggressiveness', 0.8)
            self.SPACE_CLAIMING_DISTANCE = parking_config.get('space_claiming_distance', 15.0)
            self.WALL_DETECTION_DISTANCE = parking_config.get('wall_detection_distance', 8.0)
            self.SAFE_DISTANCE_FROM_WALL = parking_config.get('safe_distance_from_wall', 3.0)
            self.PARKING_SPACE_REWARD = parking_config.get('parking_space_reward', 5.0)
            self.WALL_COLLISION_PENALTY = parking_config.get('wall_collision_penalty', -10.0)
            
            # Load parking lot bounds
            bounds_config = parking_config.get('parking_lot_bounds', {})
            self.PARKING_LOT_BOUNDS = {
                'x_min': bounds_config.get('x_min', -50),
                'x_max': bounds_config.get('x_max', 50),
                'y_min': bounds_config.get('y_min', -40),
                'y_max': bounds_config.get('y_max', 40)
            }
        else:
            # Default parking lot bounds
            self.PARKING_LOT_BOUNDS = {
                'x_min': -50, 'x_max': 50,
                'y_min': -40, 'y_max': 40
            }
    
    def act(self, obs):
        """
        Main action function for parking lot selfish behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            print("[DEBUG] No observation received")
            return 1  # IDLE if no observation
        
        # Debug: Print observation structure
        print(f"[DEBUG] Observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
        print(f"[DEBUG] Observation length: {len(obs)}")
        if len(obs) > 0:
            print(f"[DEBUG] First vehicle obs: {obs[0] if len(obs[0]) < 10 else 'Too long to print'}")
        
        # Extract ego vehicle information
        ego_info = self._extract_ego_info(obs)
        if ego_info is None:
            print("[DEBUG] Could not extract ego info")
            return 1  # IDLE if no ego vehicle info
        
        print(f"[DEBUG] Ego info: x={ego_info['x']:.2f}, y={ego_info['y']:.2f}, speed={ego_info['speed']:.2f}")
        
        # Check for wall collision avoidance (highest priority)
        wall_avoidance_action = self._check_wall_collision_avoidance(ego_info, obs)
        if wall_avoidance_action is not None:
            print(f"[DEBUG] Wall avoidance action: {wall_avoidance_action}")
            return wall_avoidance_action
        
        # Set target parking space if not already set
        if self.TARGET_PARKING_SPACE is None:
            self.TARGET_PARKING_SPACE = self._determine_target_parking_space(ego_info, obs)
            if self.TARGET_PARKING_SPACE:
                print(f"[DEBUG] Assigned target parking space: {self.TARGET_PARKING_SPACE}")
        
        # Navigate to target parking space
        navigation_action = self._navigate_to_parking_space(ego_info, obs)
        if navigation_action is not None:
            print(f"[DEBUG] Navigation action: {navigation_action}")
            return navigation_action
        
        # Apply parking-specific selfish behaviors
        parking_action = self._apply_parking_selfish_behavior(obs)
        if parking_action is not None:
            print(f"[DEBUG] Parking action: {parking_action}")
            return parking_action
        
        # Fall back to base selfish behavior
        fallback_action = super().act(obs)
        print(f"[DEBUG] Fallback action: {fallback_action}")
        return fallback_action
    
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
    
    def _extract_ego_info(self, obs):
        """
        Extract ego vehicle information from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            dict: Ego vehicle information or None
        """
        if obs is None or len(obs) == 0:
            return None
        
        try:
            # First vehicle in observation is typically the ego vehicle
            ego_obs = obs[0]
            if len(ego_obs) >= 5:  # presence, x, y, vx, vy
                return {
                    'presence': ego_obs[0],
                    'x': ego_obs[1],
                    'y': ego_obs[2],
                    'vx': ego_obs[3],
                    'vy': ego_obs[4],
                    'heading': ego_obs[5] if len(ego_obs) > 5 else 0.0,
                    'speed': math.sqrt(ego_obs[3]**2 + ego_obs[4]**2)
                }
        except (IndexError, TypeError):
            pass
        
        return None
    
    def _check_wall_collision_avoidance(self, ego_info, obs):
        """
        Check for wall collision avoidance.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            int or None: Action to avoid wall collision, None if no collision risk
        """
        if ego_info is None:
            return None
        
        # Use configurable parking lot boundaries
        PARKING_LOT_BOUNDS = self.PARKING_LOT_BOUNDS
        
        x, y = ego_info['x'], ego_info['y']
        vx, vy = ego_info['vx'], ego_info['vy']
        
        # Check proximity to walls
        wall_threats = []
        
        # Check left wall
        if x - PARKING_LOT_BOUNDS['x_min'] < self.WALL_DETECTION_DISTANCE and vx < 0:
            wall_threats.append(('left', x - PARKING_LOT_BOUNDS['x_min']))
        
        # Check right wall
        if PARKING_LOT_BOUNDS['x_max'] - x < self.WALL_DETECTION_DISTANCE and vx > 0:
            wall_threats.append(('right', PARKING_LOT_BOUNDS['x_max'] - x))
        
        # Check top wall
        if PARKING_LOT_BOUNDS['y_max'] - y < self.WALL_DETECTION_DISTANCE and vy > 0:
            wall_threats.append(('top', PARKING_LOT_BOUNDS['y_max'] - y))
        
        # Check bottom wall
        if y - PARKING_LOT_BOUNDS['y_min'] < self.WALL_DETECTION_DISTANCE and vy < 0:
            wall_threats.append(('bottom', y - PARKING_LOT_BOUNDS['y_min']))
        
        # Take evasive action if too close to walls
        if wall_threats:
            closest_wall = min(wall_threats, key=lambda x: x[1])
            wall_direction, distance = closest_wall
            
            if distance < self.SAFE_DISTANCE_FROM_WALL:
                # Emergency stop or reverse direction
                if wall_direction in ['left', 'right']:
                    return 2  # LANE_RIGHT if approaching left wall, LANE_LEFT if approaching right wall
                else:  # top or bottom
                    return 4  # SLOWER to reduce speed toward wall
        
        return None
    
    def _determine_target_parking_space(self, ego_info, obs):
        """
        Determine target parking space for the vehicle.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            dict: Target parking space information
        """
        if ego_info is None:
            return None
        
        # Simple heuristic: assign parking spaces based on vehicle position
        # In a real implementation, this would be more sophisticated
        x, y = ego_info['x'], ego_info['y']
        
        # Define parking spaces (these would ideally come from environment)
        # Use a more realistic parking lot layout
        parking_spaces = [
            {'id': 1, 'x': -40, 'y': -25, 'occupied': False},
            {'id': 2, 'x': -30, 'y': -25, 'occupied': False},
            {'id': 3, 'x': -20, 'y': -25, 'occupied': False},
            {'id': 4, 'x': -10, 'y': -25, 'occupied': False},
            {'id': 5, 'x': 0, 'y': -25, 'occupied': False},
            {'id': 6, 'x': 10, 'y': -25, 'occupied': False},
            {'id': 7, 'x': 20, 'y': -25, 'occupied': False},
            {'id': 8, 'x': 30, 'y': -25, 'occupied': False},
            {'id': 9, 'x': 40, 'y': -25, 'occupied': False},
            {'id': 10, 'x': -40, 'y': 25, 'occupied': False},
            {'id': 11, 'x': -30, 'y': 25, 'occupied': False},
            {'id': 12, 'x': -20, 'y': 25, 'occupied': False},
            {'id': 13, 'x': -10, 'y': 25, 'occupied': False},
            {'id': 14, 'x': 0, 'y': 25, 'occupied': False},
            {'id': 15, 'x': 10, 'y': 25, 'occupied': False},
            {'id': 16, 'x': 20, 'y': 25, 'occupied': False},
            {'id': 17, 'x': 30, 'y': 25, 'occupied': False},
            {'id': 18, 'x': 40, 'y': 25, 'occupied': False},
        ]
        
        # For now, assign spaces deterministically based on vehicle position
        # This ensures each vehicle gets a specific target
        if y < 0:  # Vehicle is in lower half
            # Assign to lower row parking spaces
            target_id = min(9, max(1, int((x + 50) / 10) + 1))
        else:  # Vehicle is in upper half
            # Assign to upper row parking spaces
            target_id = min(18, max(10, int((x + 50) / 10) + 10))
        
        # Find the assigned parking space
        for space in parking_spaces:
            if space['id'] == target_id:
                print(f"[DEBUG] Assigned parking space {target_id} at ({space['x']}, {space['y']})")
                return space
        
        # Fallback: find closest unoccupied parking space
        available_spaces = [space for space in parking_spaces if not space['occupied']]
        if available_spaces:
            closest_space = min(available_spaces, 
                              key=lambda space: math.sqrt((space['x'] - x)**2 + (space['y'] - y)**2))
            print(f"[DEBUG] Fallback: assigned closest space {closest_space['id']} at ({closest_space['x']}, {closest_space['y']})")
            return closest_space
        
        return None
    
    def _navigate_to_parking_space(self, ego_info, obs):
        """
        Navigate towards the target parking space.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            int or None: Navigation action, None if no navigation needed
        """
        if ego_info is None or self.TARGET_PARKING_SPACE is None:
            return None
        
        x, y = ego_info['x'], ego_info['y']
        vx, vy = ego_info['vx'], ego_info['vy']
        speed = ego_info['speed']
        target_x, target_y = self.TARGET_PARKING_SPACE['x'], self.TARGET_PARKING_SPACE['y']
        
        # Calculate distance to target
        distance_to_target = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        
        print(f"[DEBUG] Distance to target: {distance_to_target:.2f}, Current speed: {speed:.2f}")
        
        # If very close to target, slow down and try to park
        if distance_to_target < 8.0:
            if speed > 2.0:  # If moving too fast, slow down
                print("[DEBUG] Too close to target, slowing down")
                return 4  # SLOWER
            else:
                print("[DEBUG] Close to target, attempting to park")
                return 1  # IDLE to attempt parking
        
        # If moving too fast, slow down first
        if speed > 6.0:  # Parking lot speed limit
            print("[DEBUG] Moving too fast, slowing down")
            return 4  # SLOWER
        
        # Calculate direction to target
        dx = target_x - x
        dy = target_y - y
        
        # Determine if we need to change lanes or adjust speed
        if abs(dx) > abs(dy):  # Need to move horizontally
            if dx > 2.0:  # Need to move right (with threshold)
                print("[DEBUG] Moving right toward target")
                return 2  # LANE_RIGHT
            elif dx < -2.0:  # Need to move left (with threshold)
                print("[DEBUG] Moving left toward target")
                return 0  # LANE_LEFT
            else:
                # Close enough horizontally, adjust speed
                if speed < 3.0:  # Too slow
                    print("[DEBUG] Too slow, speeding up")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Good horizontal position, maintaining speed")
                    return 1  # IDLE
        else:  # Need to move vertically
            if dy > 2.0:  # Need to move up (with threshold)
                if speed < 4.0:  # Safe to speed up
                    print("[DEBUG] Moving up toward target")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Moving up but maintaining safe speed")
                    return 1  # IDLE
            elif dy < -2.0:  # Need to move down (with threshold)
                print("[DEBUG] Moving down toward target")
                return 4  # SLOWER
            else:
                # Close enough vertically, adjust speed
                if speed < 3.0:  # Too slow
                    print("[DEBUG] Too slow, speeding up")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Good vertical position, maintaining speed")
                    return 1  # IDLE
        
        return None


class ParkingLotCooperativePolicy(CooperativePolicy):
    """
    Cooperative policy with parking-specific social laws.
    
    Extends the base cooperative policy with parking assistance behaviors:
    - Targeted parking space navigation
    - Wall collision avoidance
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
        
        # Navigation and collision avoidance parameters (inherited from selfish policy)
        self.TARGET_PARKING_SPACE = None
        self.WALL_DETECTION_DISTANCE = 8.0
        self.SAFE_DISTANCE_FROM_WALL = 3.0
        self.PARKING_SPACE_REWARD = 5.0
        self.WALL_COLLISION_PENALTY = -10.0
        
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
        
        # Load parking-specific config if available
        if config and 'environment' in config:
            parking_config = config['environment'].get('parking_lot', {})
            self.WALL_DETECTION_DISTANCE = parking_config.get('wall_detection_distance', 8.0)
            self.SAFE_DISTANCE_FROM_WALL = parking_config.get('safe_distance_from_wall', 3.0)
            self.PARKING_SPACE_REWARD = parking_config.get('parking_space_reward', 5.0)
            self.WALL_COLLISION_PENALTY = parking_config.get('wall_collision_penalty', -10.0)
            
            # Load parking lot bounds
            bounds_config = parking_config.get('parking_lot_bounds', {})
            self.PARKING_LOT_BOUNDS = {
                'x_min': bounds_config.get('x_min', -50),
                'x_max': bounds_config.get('x_max', 50),
                'y_min': bounds_config.get('y_min', -40),
                'y_max': bounds_config.get('y_max', 40)
            }
        else:
            # Default parking lot bounds
            self.PARKING_LOT_BOUNDS = {
                'x_min': -50, 'x_max': 50,
                'y_min': -40, 'y_max': 40
            }
    
    def act(self, obs):
        """
        Main action function for parking lot cooperative behavior.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            print("[DEBUG] No observation received")
            return 1  # IDLE if no observation
        
        # Debug: Print observation structure
        print(f"[DEBUG] Observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
        print(f"[DEBUG] Observation length: {len(obs)}")
        if len(obs) > 0:
            print(f"[DEBUG] First vehicle obs: {obs[0] if len(obs[0]) < 10 else 'Too long to print'}")
        
        # Extract ego vehicle information
        ego_info = self._extract_ego_info(obs)
        if ego_info is None:
            print("[DEBUG] Could not extract ego info")
            return 1  # IDLE if no ego vehicle info
        
        print(f"[DEBUG] Ego info: x={ego_info['x']:.2f}, y={ego_info['y']:.2f}, speed={ego_info['speed']:.2f}")
        
        # Check for wall collision avoidance (highest priority)
        wall_avoidance_action = self._check_wall_collision_avoidance(ego_info, obs)
        if wall_avoidance_action is not None:
            print(f"[DEBUG] Wall avoidance action: {wall_avoidance_action}")
            return wall_avoidance_action
        
        # Set target parking space if not already set
        if self.TARGET_PARKING_SPACE is None:
            self.TARGET_PARKING_SPACE = self._determine_target_parking_space(ego_info, obs)
            if self.TARGET_PARKING_SPACE:
                print(f"[DEBUG] Assigned target parking space: {self.TARGET_PARKING_SPACE}")
        
        # Navigate to target parking space
        navigation_action = self._navigate_to_parking_space(ego_info, obs)
        if navigation_action is not None:
            print(f"[DEBUG] Navigation action: {navigation_action}")
            return navigation_action
        
        # Apply parking-specific cooperative behaviors
        parking_action = self._apply_parking_cooperative_behavior(obs)
        if parking_action is not None:
            print(f"[DEBUG] Parking action: {parking_action}")
            return parking_action
        
        # Fall back to base cooperative behavior
        fallback_action = super().act(obs)
        print(f"[DEBUG] Fallback action: {fallback_action}")
        return fallback_action
    
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
    
    def _extract_ego_info(self, obs):
        """
        Extract ego vehicle information from observation.
        
        Args:
            obs: Observation array
            
        Returns:
            dict: Ego vehicle information or None
        """
        if obs is None or len(obs) == 0:
            return None
        
        try:
            # First vehicle in observation is typically the ego vehicle
            ego_obs = obs[0]
            if len(ego_obs) >= 5:  # presence, x, y, vx, vy
                return {
                    'presence': ego_obs[0],
                    'x': ego_obs[1],
                    'y': ego_obs[2],
                    'vx': ego_obs[3],
                    'vy': ego_obs[4],
                    'heading': ego_obs[5] if len(ego_obs) > 5 else 0.0,
                    'speed': math.sqrt(ego_obs[3]**2 + ego_obs[4]**2)
                }
        except (IndexError, TypeError):
            pass
        
        return None
    
    def _check_wall_collision_avoidance(self, ego_info, obs):
        """
        Check for wall collision avoidance.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            int or None: Action to avoid wall collision, None if no collision risk
        """
        if ego_info is None:
            return None
        
        # Use configurable parking lot boundaries
        PARKING_LOT_BOUNDS = self.PARKING_LOT_BOUNDS
        
        x, y = ego_info['x'], ego_info['y']
        vx, vy = ego_info['vx'], ego_info['vy']
        
        # Check proximity to walls
        wall_threats = []
        
        # Check left wall
        if x - PARKING_LOT_BOUNDS['x_min'] < self.WALL_DETECTION_DISTANCE and vx < 0:
            wall_threats.append(('left', x - PARKING_LOT_BOUNDS['x_min']))
        
        # Check right wall
        if PARKING_LOT_BOUNDS['x_max'] - x < self.WALL_DETECTION_DISTANCE and vx > 0:
            wall_threats.append(('right', PARKING_LOT_BOUNDS['x_max'] - x))
        
        # Check top wall
        if PARKING_LOT_BOUNDS['y_max'] - y < self.WALL_DETECTION_DISTANCE and vy > 0:
            wall_threats.append(('top', PARKING_LOT_BOUNDS['y_max'] - y))
        
        # Check bottom wall
        if y - PARKING_LOT_BOUNDS['y_min'] < self.WALL_DETECTION_DISTANCE and vy < 0:
            wall_threats.append(('bottom', y - PARKING_LOT_BOUNDS['y_min']))
        
        # Take evasive action if too close to walls
        if wall_threats:
            closest_wall = min(wall_threats, key=lambda x: x[1])
            wall_direction, distance = closest_wall
            
            if distance < self.SAFE_DISTANCE_FROM_WALL:
                # Emergency stop or reverse direction
                if wall_direction in ['left', 'right']:
                    return 2  # LANE_RIGHT if approaching left wall, LANE_LEFT if approaching right wall
                else:  # top or bottom
                    return 4  # SLOWER to reduce speed toward wall
        
        return None
    
    def _determine_target_parking_space(self, ego_info, obs):
        """
        Determine target parking space for the vehicle.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            dict: Target parking space information
        """
        if ego_info is None:
            return None
        
        # Simple heuristic: assign parking spaces based on vehicle position
        # In a real implementation, this would be more sophisticated
        x, y = ego_info['x'], ego_info['y']
        
        # Define parking spaces (these would ideally come from environment)
        # Use a more realistic parking lot layout
        parking_spaces = [
            {'id': 1, 'x': -40, 'y': -25, 'occupied': False},
            {'id': 2, 'x': -30, 'y': -25, 'occupied': False},
            {'id': 3, 'x': -20, 'y': -25, 'occupied': False},
            {'id': 4, 'x': -10, 'y': -25, 'occupied': False},
            {'id': 5, 'x': 0, 'y': -25, 'occupied': False},
            {'id': 6, 'x': 10, 'y': -25, 'occupied': False},
            {'id': 7, 'x': 20, 'y': -25, 'occupied': False},
            {'id': 8, 'x': 30, 'y': -25, 'occupied': False},
            {'id': 9, 'x': 40, 'y': -25, 'occupied': False},
            {'id': 10, 'x': -40, 'y': 25, 'occupied': False},
            {'id': 11, 'x': -30, 'y': 25, 'occupied': False},
            {'id': 12, 'x': -20, 'y': 25, 'occupied': False},
            {'id': 13, 'x': -10, 'y': 25, 'occupied': False},
            {'id': 14, 'x': 0, 'y': 25, 'occupied': False},
            {'id': 15, 'x': 10, 'y': 25, 'occupied': False},
            {'id': 16, 'x': 20, 'y': 25, 'occupied': False},
            {'id': 17, 'x': 30, 'y': 25, 'occupied': False},
            {'id': 18, 'x': 40, 'y': 25, 'occupied': False},
        ]
        
        # For now, assign spaces deterministically based on vehicle position
        # This ensures each vehicle gets a specific target
        if y < 0:  # Vehicle is in lower half
            # Assign to lower row parking spaces
            target_id = min(9, max(1, int((x + 50) / 10) + 1))
        else:  # Vehicle is in upper half
            # Assign to upper row parking spaces
            target_id = min(18, max(10, int((x + 50) / 10) + 10))
        
        # Find the assigned parking space
        for space in parking_spaces:
            if space['id'] == target_id:
                print(f"[DEBUG] Assigned parking space {target_id} at ({space['x']}, {space['y']})")
                return space
        
        # Fallback: find closest unoccupied parking space
        available_spaces = [space for space in parking_spaces if not space['occupied']]
        if available_spaces:
            closest_space = min(available_spaces, 
                              key=lambda space: math.sqrt((space['x'] - x)**2 + (space['y'] - y)**2))
            print(f"[DEBUG] Fallback: assigned closest space {closest_space['id']} at ({closest_space['x']}, {closest_space['y']})")
            return closest_space
        
        return None
    
    def _navigate_to_parking_space(self, ego_info, obs):
        """
        Navigate towards the target parking space.
        
        Args:
            ego_info: Ego vehicle information
            obs: Observation array
            
        Returns:
            int or None: Navigation action, None if no navigation needed
        """
        if ego_info is None or self.TARGET_PARKING_SPACE is None:
            return None
        
        x, y = ego_info['x'], ego_info['y']
        vx, vy = ego_info['vx'], ego_info['vy']
        speed = ego_info['speed']
        target_x, target_y = self.TARGET_PARKING_SPACE['x'], self.TARGET_PARKING_SPACE['y']
        
        # Calculate distance to target
        distance_to_target = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        
        print(f"[DEBUG] Distance to target: {distance_to_target:.2f}, Current speed: {speed:.2f}")
        
        # If very close to target, slow down and try to park
        if distance_to_target < 8.0:
            if speed > 2.0:  # If moving too fast, slow down
                print("[DEBUG] Too close to target, slowing down")
                return 4  # SLOWER
            else:
                print("[DEBUG] Close to target, attempting to park")
                return 1  # IDLE to attempt parking
        
        # If moving too fast, slow down first
        if speed > 6.0:  # Parking lot speed limit
            print("[DEBUG] Moving too fast, slowing down")
            return 4  # SLOWER
        
        # Calculate direction to target
        dx = target_x - x
        dy = target_y - y
        
        # Determine if we need to change lanes or adjust speed
        if abs(dx) > abs(dy):  # Need to move horizontally
            if dx > 2.0:  # Need to move right (with threshold)
                print("[DEBUG] Moving right toward target")
                return 2  # LANE_RIGHT
            elif dx < -2.0:  # Need to move left (with threshold)
                print("[DEBUG] Moving left toward target")
                return 0  # LANE_LEFT
            else:
                # Close enough horizontally, adjust speed
                if speed < 3.0:  # Too slow
                    print("[DEBUG] Too slow, speeding up")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Good horizontal position, maintaining speed")
                    return 1  # IDLE
        else:  # Need to move vertically
            if dy > 2.0:  # Need to move up (with threshold)
                if speed < 4.0:  # Safe to speed up
                    print("[DEBUG] Moving up toward target")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Moving up but maintaining safe speed")
                    return 1  # IDLE
            elif dy < -2.0:  # Need to move down (with threshold)
                print("[DEBUG] Moving down toward target")
                return 4  # SLOWER
            else:
                # Close enough vertically, adjust speed
                if speed < 3.0:  # Too slow
                    print("[DEBUG] Too slow, speeding up")
                    return 3  # FASTER
                else:
                    print("[DEBUG] Good vertical position, maintaining speed")
                    return 1  # IDLE
        
        return None
