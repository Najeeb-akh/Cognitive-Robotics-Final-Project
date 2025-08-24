"""
FIXED: Simulation Scenarios Implementation (FR4)

This is the corrected version that properly integrates custom agents with highway-env.
The main fixes:
1. Proper agent-vehicle integration 
2. Correct observation-based decision making
3. Proper action application to environment
"""

import gymnasium as gym
import highway_env
import logging
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle

# Import our custom policies
from policies.selfish_policy import SelfishAgent
from policies.cooperative_policy import CooperativeAgent


class DecentralizedVehicle(ControlledVehicle):
    """
    A vehicle wrapper that uses observation-based policies instead of direct road access.
    This ensures decentralization - agents only act on local observations.
    """
    
    def __init__(self, road, position, heading=0, speed=0, policy_class=SelfishAgent, config=None):
        super().__init__(road, position, heading, speed)
        self.policy_class = policy_class
        self.config = config
        self.current_observation = None
        
    def set_observation(self, observation):
        """Set the current observation for this vehicle."""
        self.current_observation = observation
        
    def act(self, action=None):
        """
        Act based on current observation using the configured policy.
        This replaces direct road access with observation-based decision making.
        """
        if self.current_observation is None:
            return super().act()  # Fallback to default behavior
        
        # Create a temporary policy instance with observation-based behavior
        if self.policy_class == SelfishAgent:
            return self._selfish_action()
        elif self.policy_class == CooperativeAgent:
            return self._cooperative_action()
        else:
            return super().act()
    
    def _selfish_action(self):
        """Implement selfish behavior based on observation."""
        obs = self.current_observation
        
        # Simple selfish strategy: accelerate if clear ahead, change lanes if blocked
        ego_vx = obs[0][3]  # Our velocity x
        ego_vy = obs[0][4]  # Our velocity y
        
        # Check for vehicle directly ahead
        vehicle_ahead = None
        min_distance = float('inf')
        
        for i in range(1, len(obs)):
            if obs[i][0] == 1:  # Vehicle present
                rel_x = obs[i][1]  # Relative x position
                rel_y = obs[i][2]  # Relative y position
                
                # Vehicle ahead if x > 0 and in same lane (small |y|)
                if rel_x > 0 and abs(rel_y) < 2.0 and rel_x < min_distance:
                    min_distance = rel_x
                    vehicle_ahead = obs[i]
        
        # Decision logic
        if vehicle_ahead is not None and min_distance < 20.0:
            # Vehicle close ahead - try lane change or slow down
            if min_distance < 10.0:
                return 4  # SLOWER
            else:
                # Try lane change - simple heuristic
                return 0 if np.random.random() > 0.5 else 2  # LANE_LEFT or LANE_RIGHT
        else:
            # Clear ahead - accelerate
            return 3  # FASTER
    
    def _cooperative_action(self):
        """Implement cooperative behavior based on observation."""
        obs = self.current_observation
        
        # Start with selfish action
        base_action = self._selfish_action()
        
        # Add cooperative behaviors
        
        # FR3.1: Cooperative Merging
        # Look for vehicles trying to merge (in adjacent lanes moving toward our lane)
        for i in range(1, len(obs)):
            if obs[i][0] == 1:  # Vehicle present
                rel_x = obs[i][1]
                rel_y = obs[i][2]
                rel_vx = obs[i][3]
                rel_vy = obs[i][4]
                
                # Vehicle in adjacent lane moving toward our lane
                if abs(rel_y) > 2.0 and abs(rel_y) < 6.0:  # Adjacent lane
                    if ((rel_y > 0 and rel_vy < -0.5) or  # Right lane moving left
                        (rel_y < 0 and rel_vy > 0.5)):    # Left lane moving right
                        # Vehicle trying to merge - help by slowing down
                        if rel_x > -10.0 and rel_x < 20.0:  # In merge zone
                            return 4  # SLOWER
        
        # FR3.2: Polite Yielding
        # If we detect slower traffic ahead, create gaps by adjusting speed
        vehicles_ahead = []
        for i in range(1, len(obs)):
            if obs[i][0] == 1 and obs[i][1] > 0 and abs(obs[i][2]) < 2.0:
                vehicles_ahead.append(obs[i])
        
        if vehicles_ahead:
            # Sort by distance
            vehicles_ahead.sort(key=lambda v: v[1])
            closest_ahead = vehicles_ahead[0]
            
            # If multiple vehicles ahead are close, increase following distance
            if len(vehicles_ahead) > 1 and closest_ahead[1] < 15.0:
                return 4  # SLOWER - create more space
        
        # FR3.3: Phantom Jam Mitigation
        # Count nearby vehicles to estimate density
        nearby_vehicles = sum(1 for i in range(1, len(obs)) 
                            if obs[i][0] == 1 and abs(obs[i][1]) < 50.0)
        
        if nearby_vehicles > 8:  # High density threshold
            # In dense traffic, be more conservative
            if base_action == 3:  # If we were going to accelerate
                return 1  # IDLE instead - maintain current speed
        
        return base_action


def create_highway_scenario(config, render_mode='rgb_array'):
    """
    FIXED: Create Scenario A with proper agent integration.
    """
    env_config = config.get('environment', {}).get('highway', {})
    
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100], 
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": env_config.get('lanes_count', 4),
        "vehicles_count": env_config.get('vehicles_count', 50),
        "controlled_vehicles": 1,  # Only ego vehicle is controlled
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 40),
        "ego_spacing": 2,
        "vehicles_density": 1,
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False
    }
    
    env = gym.make('highway-v0', render_mode=render_mode)
    env.unwrapped.configure(highway_config)
    return env


def create_merge_scenario(config, render_mode='rgb_array'):
    """
    FIXED: Create Scenario B with proper agent integration.
    """
    env_config = config.get('environment', {}).get('merge', {})
    
    merge_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20], 
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": env_config.get('lanes_count', 3),
        "vehicles_count": env_config.get('vehicles_count', 40), 
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 40),
        "ego_spacing": 2,
        "vehicles_density": 1,
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False
    }
    
    env = gym.make('merge-v0', render_mode=render_mode)
    env.unwrapped.configure(merge_config)
    return env


def populate_environment_with_agents(env, agent_composition, config):
    """
    FIXED: Properly integrate custom agents with vehicles.
    
    This version creates a policy mapping that determines how the ego vehicle
    should behave based on the agent composition.
    """
    env.reset()
    
    # Determine ego vehicle behavior based on composition
    selfish_ratio = agent_composition.get('selfish_ratio', 0.5)
    
    # Handle different composition types
    if selfish_ratio == 1.0:
        ego_policy = SelfishAgent
        behavior_type = "selfish"
    elif selfish_ratio == 0.0:
        ego_policy = CooperativeAgent  
        behavior_type = "cooperative"
    elif selfish_ratio == 0.5:
        # Mixed composition: randomly choose for each run
        import random
        if random.random() < 0.5:
            ego_policy = SelfishAgent
            behavior_type = "mixed_selfish"
        else:
            ego_policy = CooperativeAgent
            behavior_type = "mixed_cooperative"
    else:
        # Default to majority type for other ratios
        if selfish_ratio > 0.5:
            ego_policy = SelfishAgent
            behavior_type = "selfish"
        else:
            ego_policy = CooperativeAgent  
            behavior_type = "cooperative"
    
    # Create a decentralized vehicle for the ego
    road = env.unwrapped.road
    ego_vehicle = road.vehicles[0]  # First vehicle is ego
    
    # Replace with our decentralized version
    decentralized_ego = DecentralizedVehicle(
        road=ego_vehicle.road,
        position=ego_vehicle.position,
        heading=ego_vehicle.heading,
        speed=ego_vehicle.speed,
        policy_class=ego_policy,
        config=config
    )
    
    # Replace in road
    road.vehicles[0] = decentralized_ego
    
    logging.info(f"Configured ego vehicle with {behavior_type} policy")
    logging.info(f"Environment has {len(road.vehicles)} total vehicles")
    
    return [decentralized_ego]  # Return controlled agents


def get_scenario_configurations():
    """
    Standard scenario configurations for testing.
    """
    scenarios = [
        ("Highway_100_Selfish", create_highway_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Highway_100_Cooperative", create_highway_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Highway_50_50_Mix", create_highway_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        ("Merge_100_Selfish", create_merge_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Merge_100_Cooperative", create_merge_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Merge_50_50_Mix", create_merge_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
    ]
    
    return scenarios