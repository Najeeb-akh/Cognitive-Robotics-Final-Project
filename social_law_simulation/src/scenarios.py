"""
Simulation Scenarios Implementation (FR4)

This module creates and configures the two highway-env scenarios:
- Scenario A: Highway Driving (multi-lane straight highway)
- Scenario B: Merge Ramp (highway with merging lane)

Each scenario is configured to test different aspects of cooperative behavior.
"""

import gymnasium as gym
import highway_env
import logging
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv


def create_highway_scenario(config, render_mode='rgb_array'):
    """
    Create Scenario A: Highway Driving
    
    A multi-lane straight highway with continuous vehicle inflow.
    Tests general traffic flow and lane-changing behaviors.
    
    Args:
        config (dict): Configuration dictionary
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured highway environment
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('highway', {})
    
    # Create highway environment with config
    highway_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # Number of vehicles to observe
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": True  # Add normalization to fix observation space warnings
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": env_config.get('lanes_count', 4),
        "vehicles_count": env_config.get('vehicles_count', 50),
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
        "offroad_terminal": False,
        "terminate_on_collision": False
    }
    
    # Create environment and configure it
    env = gym.make('highway-v0', render_mode=render_mode)
    env.unwrapped.configure(highway_config)
    return env


def create_merge_scenario(config, render_mode='rgb_array'):
    """
    Create Scenario B: Merge Ramp
    
    A highway with a lane merging into it.
    Critical for testing Cooperative Merging and Polite Yielding social laws.
    
    Args:
        config (dict): Configuration dictionary
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured merge environment
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('merge', {})
    
    # Create merge environment with config  
    merge_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,  # Number of vehicles to observe
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": True  # Add normalization to fix observation space warnings
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
        "offroad_terminal": False,
        "terminate_on_collision": False,
        "merging_lane_spawn_probability": 0.2,  # Probability of spawning in merge lane
        "merging_speed_ratio": 0.8  # Speed ratio for merging vehicles
    }
    
    # Create environment and configure it
    env = gym.make('merge-v0', render_mode=render_mode)
    env.unwrapped.configure(merge_config)
    return env


def populate_environment_with_agents(env, agent_composition, config):
    """
    DEPRECATED: This function is no longer needed with the new policy-based architecture.
    
    The new architecture uses policy classes (SelfishPolicy, CooperativePolicy) 
    that work directly with highway-env's standard gymnasium interface instead
    of replacing vehicle objects.
    
    This function is kept for compatibility but does nothing.
    
    Args:
        env (gym.Env): Environment (unused)
        agent_composition (dict): Dict with ratios (unused)
        config (dict): Configuration dictionary (unused)
        
    Returns:
        list: Empty list (no agents needed)
    """
    logging.warning("populate_environment_with_agents() is deprecated - using policy-based architecture")
    return []


def get_scenario_configurations():
    """
    Get the standard scenario and composition configurations for FR4.
    
    Returns:
        list: List of (scenario_name, scenario_func, composition) tuples
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


