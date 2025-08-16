"""
Extended Simulation Scenarios Implementation

This module extends the original scenarios.py with additional complex traffic scenarios:
- Intersection: Multi-directional traffic with turn conflicts  
- Roundabout: Circular traffic pattern with entry/exit coordination
- Racetrack: High-speed continuous loop with overtaking challenges

All original functionality from scenarios.py is preserved and imported unchanged.
"""

import gymnasium as gym
import highway_env
import logging
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv

# Import all original scenarios unchanged
from scenarios import get_scenario_configurations, create_highway_scenario, create_merge_scenario


def create_intersection_scenario(config, render_mode='rgb_array'):
    """
    Create Intersection Scenario: Multi-directional traffic coordination
    
    A 4-way intersection with traffic flows from all directions requiring
    complex right-of-way negotiations and turn coordination.
    Tests social laws for gap provision, turn-taking, and adaptive yielding.
    
    Args:
        config (dict): Configuration dictionary
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured intersection environment
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('intersection', {})
    
    # Create intersection environment configuration
    intersection_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 20,  # More vehicles needed for 4-way traffic
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-150, 150],  # Larger observation range for intersection
                "y": [-150, 150],
                "vx": [-25, 25],
                "vy": [-25, 25]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,  # 4-way intersection
        "vehicles_count": env_config.get('vehicles_count', 60),  # Higher density for complexity
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 50),  # Longer episodes for complex interactions
        "ego_spacing": 2,
        "vehicles_density": 1.2,  # Higher density at intersection
        "collision_reward": -2,  # Higher penalty for intersection collisions
        "right_lane_reward": 0.05,
        "high_speed_reward": 0.3,  # Lower speed emphasis for safety
        "lane_change_reward": -0.1,  # Discourage frequent lane changes
        "reward_speed_range": [15, 25],  # Lower speed range for intersection
        "normalize_reward": True,
        "offroad_terminal": False,
        "terminate_on_collision": False,
        # Intersection-specific parameters
        "intersection_size": env_config.get('intersection_size', 20),
        "traffic_light_timing": env_config.get('traffic_light_timing', 30),
        "approach_speed_limit": env_config.get('approach_speed_limit', 50),
        "turn_lane_length": env_config.get('turn_lane_length', 50)
    }
    
    # Use intersection-v0 environment if available, otherwise adapt highway-v0
    try:
        env = gym.make('intersection-v0', render_mode=render_mode)
        env.unwrapped.configure(intersection_config)
        logging.info("Created intersection-v0 environment")
    except Exception as e:
        # Fallback to highway-v0 with intersection-like configuration
        logging.warning(f"intersection-v0 not available ({e}), using modified highway-v0")
        env = gym.make('highway-v0', render_mode=render_mode)
        env.unwrapped.configure(intersection_config)
    
    return env


def create_roundabout_scenario(config, render_mode='rgb_array'):
    """
    Create Roundabout Scenario: Circular traffic coordination
    
    A multi-entry roundabout with yielding protocols and exit coordination.
    Tests social laws for entry facilitation, flow maintenance, and exit courtesy.
    
    Args:
        config (dict): Configuration dictionary  
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured roundabout environment
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('roundabout', {})
    
    # Create roundabout environment configuration
    roundabout_config = {
        "observation": {
            "type": "Kinematics", 
            "vehicles_count": 18,  # Moderate number for roundabout complexity
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-120, 120],  # Circular observation range
                "y": [-120, 120],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted", 
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": env_config.get('lane_count', 2),  # Inner/outer lanes
        "vehicles_count": env_config.get('vehicles_count', 45),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 45),
        "ego_spacing": 2,
        "vehicles_density": 1.1,
        "collision_reward": -1.5,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.35,
        "lane_change_reward": -0.05,  # Small penalty for lane changes
        "reward_speed_range": [20, 30],  # Moderate speed for roundabout safety
        "normalize_reward": True,
        "offroad_terminal": False,
        "terminate_on_collision": False,
        # Roundabout-specific parameters
        "radius": env_config.get('radius', 50),
        "entry_points": env_config.get('entry_points', 4),
        "entry_yield_distance": env_config.get('entry_yield_distance', 20),
        "circulating_speed_limit": env_config.get('circulating_speed_limit', 25),
        "entry_acceleration_lane": env_config.get('entry_acceleration_lane', 15)
    }
    
    # Use roundabout-v0 environment if available, otherwise adapt highway-v0
    try:
        env = gym.make('roundabout-v0', render_mode=render_mode)
        env.unwrapped.configure(roundabout_config)
        logging.info("Created roundabout-v0 environment")
    except Exception as e:
        # Fallback to highway-v0 with roundabout-like configuration
        logging.warning(f"roundabout-v0 not available ({e}), using modified highway-v0")
        env = gym.make('highway-v0', render_mode=render_mode)
        env.unwrapped.configure(roundabout_config)
    
    return env


def create_racetrack_scenario(config, render_mode='rgb_array'):
    """
    Create Racetrack Scenario: High-speed overtaking coordination
    
    A continuous loop with high-speed traffic requiring safe overtaking,
    defensive positioning, and cooperative slipstreaming behaviors.
    Tests social laws for overtaking protocols and speed coordination.
    
    Args:
        config (dict): Configuration dictionary
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured racetrack environment  
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('racetrack', {})
    
    # Create racetrack environment configuration
    racetrack_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,  # Fewer vehicles for high-speed safety
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-200, 200],  # Extended range for high speeds
                "y": [-100, 100],
                "vx": [-40, 40],   # Higher speed range
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": True
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": env_config.get('lane_count', 3),  # Racing lanes
        "vehicles_count": env_config.get('vehicles_count', 25),  # Lower density for high speeds
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None), 
        "duration": env_config.get('duration', 60),  # Longer for racing dynamics
        "ego_spacing": 3,  # Increased spacing for high speeds
        "vehicles_density": 0.8,  # Lower density for safety
        "collision_reward": -3,  # Very high penalty for high-speed collisions
        "right_lane_reward": 0.0,  # No lane preference on racetrack
        "high_speed_reward": 0.6,  # High emphasis on speed
        "lane_change_reward": 0.1,  # Reward tactical lane changes
        "reward_speed_range": [25, 40],  # High speed range
        "normalize_reward": True,
        "offroad_terminal": False,
        "terminate_on_collision": False,
        # Racetrack-specific parameters
        "track_length": env_config.get('track_length', 2000),
        "corner_count": env_config.get('corner_count', 4),
        "straight_length": env_config.get('straight_length', 400),
        "speed_limit": env_config.get('speed_limit', 120),
        "slipstream_effect": env_config.get('slipstream_effect', True)
    }
    
    # Use racetrack-v0 environment if available, otherwise adapt highway-v0
    try:
        env = gym.make('racetrack-v0', render_mode=render_mode)
        env.unwrapped.configure(racetrack_config)
        logging.info("Created racetrack-v0 environment")
    except Exception as e:
        # Fallback to highway-v0 with racetrack-like configuration
        logging.warning(f"racetrack-v0 not available ({e}), using modified highway-v0")
        env = gym.make('highway-v0', render_mode=render_mode)
        env.unwrapped.configure(racetrack_config)
    
    return env


def get_extended_scenario_configurations():
    """
    Get extended scenario and composition configurations.
    
    Includes all original scenarios plus new intersection, roundabout, and racetrack scenarios.
    Original scenarios remain completely unchanged and are imported as-is.
    
    Returns:
        list: List of (scenario_name, scenario_func, composition) tuples
    """
    # Get all original scenarios unchanged
    original_scenarios = get_scenario_configurations()
    
    # Define new extended scenarios
    new_scenarios = [
        # Intersection scenarios
        ("Intersection_100_Selfish", create_intersection_scenario, 
         {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Intersection_100_Cooperative", create_intersection_scenario, 
         {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Intersection_50_50_Mix", create_intersection_scenario, 
         {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        
        # Roundabout scenarios  
        ("Roundabout_100_Selfish", create_roundabout_scenario,
         {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Roundabout_100_Cooperative", create_roundabout_scenario,
         {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Roundabout_50_50_Mix", create_roundabout_scenario,
         {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        
        # Racetrack scenarios
        ("Racetrack_100_Selfish", create_racetrack_scenario,
         {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Racetrack_100_Cooperative", create_racetrack_scenario,
         {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Racetrack_50_50_Mix", create_racetrack_scenario,
         {"selfish_ratio": 0.5, "cooperative_ratio": 0.5})
    ]
    
    # Return combined list: original + new scenarios
    return original_scenarios + new_scenarios


def get_original_scenarios_only():
    """
    Get only the original scenarios (for regression testing).
    
    Returns:
        list: Original scenario configurations only
    """
    return get_scenario_configurations()


def get_new_scenarios_only():
    """
    Get only the new extended scenarios.
    
    Returns:
        list: New scenario configurations only  
    """
    all_scenarios = get_extended_scenario_configurations()
    original_count = len(get_scenario_configurations())
    return all_scenarios[original_count:]  # Return only new scenarios