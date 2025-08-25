"""
Racetrack-Specific Policy Implementation

This module implements enhanced cooperative and competitive behaviors specifically 
designed for racetrack scenarios, focusing on high-speed coordination, safe 
overtaking, and cooperative slipstreaming.
"""

import numpy as np
import logging
import math
from .cooperative_policy import CooperativePolicy


class RacetrackCooperativePolicy(CooperativePolicy):
    """
    Enhanced cooperative policy for racetrack scenarios.
    
    Implements racetrack-specific social laws:
    - Safe Overtaking Protocol: Coordinates safe high-speed overtaking
    - Defensive Positioning: Allows faster vehicles to pass safely
    - Slipstream Cooperation: Enables mutual benefit from drafting
    """
    
    def __init__(self, config=None):
        """Initialize racetrack-specific cooperative policy."""
        super().__init__(config)
        
        # Racetrack-specific social law parameters
        if config and 'social_laws' in config:
            social_laws = config['social_laws']
            
            # Safe Overtaking Protocol parameters
            if 'safe_overtaking_protocol' in social_laws:
                overtake_config = social_laws['safe_overtaking_protocol']
                self.minimum_speed_differential = overtake_config.get('minimum_speed_differential', 10)
                self.safe_overtaking_distance = overtake_config.get('safe_overtaking_distance', 50)
                self.abort_threshold = overtake_config.get('abort_threshold', 20)
                self.safe_overtaking_enabled = overtake_config.get('enabled', True)
            else:
                self.minimum_speed_differential = 10
                self.safe_overtaking_distance = 50
                self.abort_threshold = 20
                self.safe_overtaking_enabled = True
                
            # Defensive Positioning parameters
            if 'defensive_positioning' in social_laws:
                defense_config = social_laws['defensive_positioning']
                self.allow_faster_pass = defense_config.get('allow_faster_pass', True)
                self.speed_differential_threshold = defense_config.get('speed_differential_threshold', 15)
                self.defensive_gap_size = defense_config.get('defensive_gap_size', 20)
                self.defensive_positioning_enabled = defense_config.get('enabled', True)
            else:
                self.allow_faster_pass = True
                self.speed_differential_threshold = 15
                self.defensive_gap_size = 20
                self.defensive_positioning_enabled = True
                
            # Slipstream Cooperation parameters
            if 'slipstream_cooperation' in social_laws:
                slipstream_config = social_laws['slipstream_cooperation']
                self.cooperation_speed_range = slipstream_config.get('cooperation_speed_range', [80, 120])
                self.lead_consistency_factor = slipstream_config.get('lead_consistency_factor', 0.95)
                self.alternating_lead_distance = slipstream_config.get('alternating_lead_distance', 500)
                self.slipstream_cooperation_enabled = slipstream_config.get('enabled', True)
            else:
                self.cooperation_speed_range = [80, 120]
                self.lead_consistency_factor = 0.95
                self.alternating_lead_distance = 500
                self.slipstream_cooperation_enabled = True
        else:
            # Default parameters
            self.minimum_speed_differential = 10
            self.safe_overtaking_distance = 50
            self.abort_threshold = 20
            self.safe_overtaking_enabled = True
            self.allow_faster_pass = True
            self.speed_differential_threshold = 15
            self.defensive_gap_size = 20
            self.defensive_positioning_enabled = True
            self.cooperation_speed_range = [80, 120]
            self.lead_consistency_factor = 0.95
            self.alternating_lead_distance = 500
            self.slipstream_cooperation_enabled = True
        
        # Internal state for racetrack dynamics
        self.overtaking_state = 'none'  # 'none', 'preparing', 'executing', 'completing'
        self.slipstream_partner = None
        self.lead_distance_traveled = 0
        self.last_overtake_attempt = 0
        
        logging.info("RacetrackCooperativePolicy initialized with high-speed social laws")
    
    def act(self, obs):
        """
        Enhanced action selection with racetrack-specific cooperative behaviors.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return super().act(obs)
        
        # Analyze racetrack context
        racetrack_context = self._analyze_racetrack_context(obs)
        
        # Update internal racetrack state
        self._update_racetrack_state(racetrack_context)
        
        # Check for racetrack-specific social law opportunities
        if self._should_execute_safe_overtaking(racetrack_context):
            return self._execute_safe_overtaking_protocol(obs, racetrack_context)
        
        if self._should_apply_defensive_positioning(racetrack_context):
            return self._execute_defensive_positioning(obs, racetrack_context)
        
        if self._should_cooperate_in_slipstream(racetrack_context):
            return self._execute_slipstream_cooperation(obs, racetrack_context)
        
        # High-speed racetrack navigation
        return self._navigate_racetrack(obs, racetrack_context)
    
    def _analyze_racetrack_context(self, obs):
        """
        Analyze the racetrack racing context.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            dict: Context information about racetrack racing
        """
        context = {
            'ego_speed': 0,
            'ego_position': None,
            'front_vehicle': None,
            'rear_vehicle': None,
            'overtaking_opportunities': [],
            'faster_approaching': [],
            'slipstream_candidates': [],
            'track_position': 'straight'  # 'straight', 'corner'
        }
        
        if len(obs) == 0:
            return context
        
        # Ego vehicle information
        ego_info = obs[0]
        context['ego_speed'] = abs(ego_info[3]) if len(ego_info) > 3 else 30  # Use vx as speed
        context['ego_position'] = {
            'x': ego_info[1] if len(ego_info) > 1 else 0,
            'y': ego_info[2] if len(ego_info) > 2 else 0
        }
        
        # Classify track position (simplified heuristic)
        ego_y = context['ego_position']['y']
        if abs(ego_y) > 20:  # Assuming corners have higher y values
            context['track_position'] = 'corner'
        
        # Analyze other vehicles
        front_distance = float('inf')
        rear_distance = float('inf')
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1:  # Vehicle present
                rel_x = vehicle[1]
                rel_y = vehicle[2] 
                rel_speed = vehicle[3] if len(vehicle) > 3 else 30
                vehicle_speed = context['ego_speed'] + rel_speed  # Approximate absolute speed
                
                distance = abs(rel_x)
                
                vehicle_info = {
                    'index': i,
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'speed': vehicle_speed,
                    'distance': distance,
                    'speed_differential': vehicle_speed - context['ego_speed']
                }
                
                if rel_x > 0 and distance < front_distance:  # Vehicle ahead
                    front_distance = distance
                    context['front_vehicle'] = vehicle_info
                elif rel_x < 0 and distance < rear_distance:  # Vehicle behind
                    rear_distance = distance
                    context['rear_vehicle'] = vehicle_info
                
                # Categorize vehicles for different social laws
                if rel_x > 0 and distance < self.safe_overtaking_distance:
                    # Potential overtaking target
                    if vehicle_info['speed_differential'] < -self.minimum_speed_differential:
                        context['overtaking_opportunities'].append(vehicle_info)
                
                if rel_x < 0 and vehicle_info['speed_differential'] > self.speed_differential_threshold:
                    # Faster vehicle approaching from behind
                    context['faster_approaching'].append(vehicle_info)
                
                if (self.cooperation_speed_range[0] <= context['ego_speed'] <= self.cooperation_speed_range[1] and
                    self.cooperation_speed_range[0] <= vehicle_speed <= self.cooperation_speed_range[1]):
                    # Potential slipstream partner
                    if 5 < distance < 30:  # Optimal slipstream range
                        context['slipstream_candidates'].append(vehicle_info)
        
        return context
    
    def _should_execute_safe_overtaking(self, context):
        """
        Determine if safe overtaking should be executed.
        
        Args:
            context: Racetrack context
            
        Returns:
            bool: True if safe overtaking is beneficial and possible
        """
        if not self.safe_overtaking_enabled:
            return False
        
        # Don't overtake in corners - too dangerous
        if context['track_position'] == 'corner':
            return False
        
        # Cooldown after last overtake attempt
        if self.last_overtake_attempt > 0:
            self.last_overtake_attempt -= 1
            return False
        
        # Check for overtaking opportunities
        return len(context['overtaking_opportunities']) > 0
    
    def _should_apply_defensive_positioning(self, context):
        """
        Determine if defensive positioning should be applied.
        
        Args:
            context: Racetrack context
            
        Returns:
            bool: True if defensive positioning is needed
        """
        if not self.defensive_positioning_enabled or not self.allow_faster_pass:
            return False
        
        # Check for faster vehicles approaching
        return len(context['faster_approaching']) > 0
    
    def _should_cooperate_in_slipstream(self, context):
        """
        Determine if slipstream cooperation is beneficial.
        
        Args:
            context: Racetrack context
            
        Returns:
            bool: True if slipstream cooperation should be engaged
        """
        if not self.slipstream_cooperation_enabled:
            return False
        
        # Only in speed range suitable for slipstreaming
        if not (self.cooperation_speed_range[0] <= context['ego_speed'] <= self.cooperation_speed_range[1]):
            return False
        
        # Check for suitable slipstream candidates
        return len(context['slipstream_candidates']) > 0
    
    def _execute_safe_overtaking_protocol(self, obs, context):
        """
        Execute safe overtaking maneuver.
        
        Args:
            obs: Highway-env observation array
            context: Racetrack context
            
        Returns:
            int: Action for safe overtaking
        """
        target = context['overtaking_opportunities'][0]  # Take first opportunity
        
        if self.overtaking_state == 'none':
            # Start preparing overtaking maneuver
            self.overtaking_state = 'preparing'
            # Check if clear lane available for overtaking
            if abs(target['rel_y']) < 2:  # Same lane - need to change lanes
                if target['rel_y'] > 0:  # Target in left, go right
                    logging.debug("Initiating safe overtaking - moving right")
                    return 2  # LANE_RIGHT
                else:  # Target in right, go left
                    logging.debug("Initiating safe overtaking - moving left")
                    return 0  # LANE_LEFT
        
        elif self.overtaking_state == 'preparing':
            # Accelerate to complete overtaking
            self.overtaking_state = 'executing'
            logging.debug("Executing safe overtaking - accelerating")
            return 3  # FASTER
        
        elif self.overtaking_state == 'executing':
            # Check if overtaking is complete
            if target['rel_x'] < 0:  # Target is now behind
                self.overtaking_state = 'completing'
                # Move back to optimal racing line if needed
                return 1  # IDLE - completed overtake
            else:
                # Continue accelerating
                return 3  # FASTER
        
        elif self.overtaking_state == 'completing':
            # Reset state and set cooldown
            self.overtaking_state = 'none'
            self.last_overtake_attempt = 20  # Cooldown period
            return 1  # IDLE
        
        return 1  # Default IDLE
    
    def _execute_defensive_positioning(self, obs, context):
        """
        Execute defensive positioning to allow faster vehicles to pass.
        
        Args:
            obs: Highway-env observation array  
            context: Racetrack context
            
        Returns:
            int: Action for defensive positioning
        """
        faster_vehicle = context['faster_approaching'][0]
        
        # Move to provide gap for faster vehicle
        if abs(faster_vehicle['rel_y']) < 2:  # Same lane
            # Move to side to provide passing opportunity
            if faster_vehicle['rel_y'] > 0:  # Faster vehicle on left, move right
                logging.debug("Defensive positioning - moving right for faster vehicle")
                return 2  # LANE_RIGHT
            else:  # Faster vehicle on right, move left  
                logging.debug("Defensive positioning - moving left for faster vehicle")
                return 0  # LANE_LEFT
        else:
            # Different lanes - maintain position but ensure adequate gap
            if faster_vehicle['distance'] < self.defensive_gap_size:
                logging.debug("Defensive positioning - slowing to provide gap")
                return 4  # SLOWER to provide gap
        
        return 1  # IDLE
    
    def _execute_slipstream_cooperation(self, obs, context):
        """
        Execute slipstream cooperation behavior.
        
        Args:
            obs: Highway-env observation array
            context: Racetrack context
            
        Returns:
            int: Action for slipstream cooperation
        """
        partner = context['slipstream_candidates'][0]
        
        # Determine if we should lead or follow
        if partner['rel_x'] > 0:  # Partner ahead - we follow
            target_distance = 8  # Optimal slipstream distance
            current_distance = partner['distance']
            
            if current_distance > target_distance * 1.2:
                logging.debug("Slipstream cooperation - closing distance to partner")
                return 3  # FASTER to close gap
            elif current_distance < target_distance * 0.8:
                logging.debug("Slipstream cooperation - maintaining safe distance")
                return 4  # SLOWER to maintain safe distance
            else:
                # Optimal slipstream position - maintain speed consistency
                return 1  # IDLE
        
        else:  # Partner behind - we lead
            # Maintain consistent speed for partner's slipstream benefit
            target_speed = np.mean(self.cooperation_speed_range)
            
            if context['ego_speed'] < target_speed * 0.9:
                return 3  # FASTER to reach target speed
            elif context['ego_speed'] > target_speed * 1.1:
                return 4  # SLOWER to reach target speed
            else:
                logging.debug("Slipstream cooperation - maintaining consistent lead speed")
                return 1  # IDLE - consistent leading
    
    def _navigate_racetrack(self, obs, context):
        """
        Navigate the racetrack with high-speed optimization.
        
        Args:
            obs: Highway-env observation array
            context: Racetrack context
            
        Returns:
            int: Navigation action
        """
        # High-speed navigation logic
        target_speed = 35  # High target speed for racetrack
        
        if context['front_vehicle'] is not None:
            front_distance = context['front_vehicle']['distance']
            
            if front_distance < 20:  # Too close at high speed
                return 4  # SLOWER for safety
            elif front_distance > 40 and context['ego_speed'] < target_speed:
                return 3  # FASTER to optimize speed
        else:
            # No vehicle ahead - optimize for speed
            if context['ego_speed'] < target_speed:
                return 3  # FASTER
        
        # Corner handling
        if context['track_position'] == 'corner':
            if context['ego_speed'] > 25:  # Reduce speed for corners
                return 4  # SLOWER
        
        return 1  # IDLE - maintain current racing speed
    
    def _update_racetrack_state(self, context):
        """
        Update internal racetrack racing state.
        
        Args:
            context: Racetrack context
        """
        # Update lead distance for slipstream cooperation
        if self.slipstream_partner:
            self.lead_distance_traveled += context['ego_speed'] * 0.1  # Approximate distance increment
        
        # Reset partner if no longer in slipstream range
        if not context['slipstream_candidates']:
            self.slipstream_partner = None
            self.lead_distance_traveled = 0


class RacetrackSelfishPolicy:
    """
    Selfish policy specialized for racetrack scenarios.
    
    Maximizes individual performance through aggressive racing tactics
    while maintaining basic safety.
    """
    
    def __init__(self, config=None):
        """Initialize racetrack-specific selfish policy."""
        from .selfish_policy import SelfishPolicy
        self.base_policy = SelfishPolicy(config)
        
        # Racetrack-specific parameters (more aggressive)
        self.aggressive_overtaking = True
        self.minimal_gap_acceptance = 0.3  # Accept smaller gaps
        self.high_speed_preference = True  # Prioritize speed over cooperation
        self.blocking_behavior = True      # Block other vehicles' overtaking attempts
        
        # Racing-specific thresholds
        self.overtaking_threshold = 5      # Lower speed difference needed
        self.defensive_gap_size = 10       # Smaller gaps provided
        
        logging.info("RacetrackSelfishPolicy initialized with aggressive racing parameters")
    
    def act(self, obs):
        """
        Action selection with aggressive racetrack racing tactics.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            int: Action index for highway-env
        """
        if obs is None or len(obs) == 0:
            return self.base_policy.act(obs)
        
        # Analyze for aggressive opportunities
        ego_speed = abs(obs[0][3]) if len(obs[0]) > 3 else 30
        
        # Very aggressive speed optimization
        target_speed = 40  # Higher target speed than cooperative
        
        front_vehicle = self._find_front_vehicle_racing(obs)
        
        if front_vehicle:
            front_distance = abs(front_vehicle[1])
            front_speed = ego_speed + front_vehicle[3]  # Approximate
            
            # Aggressive overtaking behavior
            if (ego_speed > front_speed + self.overtaking_threshold and 
                front_distance > 15):  # Smaller safety margin
                # Attempt aggressive lane change for overtaking
                if front_vehicle[2] > 0:  # Front vehicle on left
                    return 2  # LANE_RIGHT
                else:  # Front vehicle on right or center
                    return 0  # LANE_LEFT
            
            # Aggressive following - closer distances
            if front_distance < 15:  # Closer following than cooperative
                return 4  # SLOWER
            elif front_distance > 25 and ego_speed < target_speed:
                return 3  # FASTER
        else:
            # No vehicle ahead - maximum speed
            if ego_speed < target_speed:
                return 3  # FASTER
        
        return 1  # IDLE
    
    def _find_front_vehicle_racing(self, obs):
        """
        Find front vehicle with racing-specific logic.
        
        Args:
            obs: Highway-env observation array
            
        Returns:
            array or None: Front vehicle for racing context
        """
        closest_front = None
        min_distance = float('inf')
        
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] == 1 and vehicle[1] > 0:  # Present and ahead
                distance = abs(vehicle[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_front = vehicle
        
        return closest_front