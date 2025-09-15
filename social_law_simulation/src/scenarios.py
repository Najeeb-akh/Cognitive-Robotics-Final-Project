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
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False  # Use physical units so policy thresholds apply
        },
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"],
        },
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": env_config.get('lanes_count', 4),
        "vehicles_count": env_config.get('vehicles_count', 50),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 200),  # Increased from 40 to allow longer episodes
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False,
        # Rely on default collision termination behavior from highway-env
    }
    
    # Create environment and configure it (prefer v1, fallback to v0)
    try:
        env = gym.make('highway-v1', render_mode=render_mode)
        env.unwrapped.configure(highway_config)
        logging.info("Created highway-v1 environment")
    except Exception as e:
        logging.warning(f"highway-v1 not available ({e}), using highway-v0")
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
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False  # Use physical units so policy thresholds apply
        },
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"],
        },
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": env_config.get('lanes_count', 3),
        "vehicles_count": env_config.get('vehicles_count', 40),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 200),  # Increased from 40 to allow longer episodes
        "collision_reward": -1,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False,
        # Keep standard frequencies unless overridden in env defaults/config
        # "simulation_frequency": 15,
        # "policy_frequency": 3,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        # Rendering options managed externally
    }
    
    # Create environment and configure it (prefer v1, fallback to v0)
    try:
        env = gym.make('merge-v1', render_mode=render_mode)
        env.unwrapped.configure(merge_config)
        logging.info("Created merge-v1 environment")
    except Exception as e:
        logging.warning(f"merge-v1 not available ({e}), using merge-v0")
        env = gym.make('merge-v0', render_mode=render_mode)
        env.unwrapped.configure(merge_config)
    return env


def create_intersection_scenario(config, render_mode='rgb_array'):
    """
    Create Intersection Scenario: Multi-directional traffic coordination
    """
    env_config = config.get('environment', {}).get('intersection', {})
    intersection_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 20,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {"x": [-150, 150], "y": [-150, 150], "vx": [-25, 25], "vy": [-25, 25]},
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False
        },
        "action": {"type": "DiscreteMetaAction",
                    "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]},
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": 4,
        "vehicles_count": env_config.get('vehicles_count', 60),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 50),
        "ego_spacing": 2,
        "collision_reward": -2,
        "right_lane_reward": 0.05,
        "high_speed_reward": 0.3,
        "lane_change_reward": -0.1,
        "reward_speed_range": [15, 25],
        "normalize_reward": True,
        "offroad_terminal": False,
    }
    try:
        env = gym.make('intersection-v1', render_mode=render_mode)
        env.unwrapped.configure(intersection_config)
        logging.info("Created intersection-v1 environment")
    except Exception as e:
        logging.warning(f"intersection-v1 not available ({e}), falling back to v0")
        try:
            env = gym.make('intersection-v0', render_mode=render_mode)
            env.unwrapped.configure(intersection_config)
            logging.info("Created intersection-v0 environment")
        except Exception as e2:
            logging.warning(f"intersection-v0 not available ({e2}), using modified highway-v0")
            env = gym.make('highway-v0', render_mode=render_mode)
            env.unwrapped.configure(intersection_config)
    return env


def create_roundabout_scenario(config, render_mode='rgb_array'):
    """
    Create Roundabout Scenario: Circular traffic coordination
    """
    env_config = config.get('environment', {}).get('roundabout', {})
    roundabout_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 18,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {"x": [-120, 120], "y": [-120, 120], "vx": [-20, 20], "vy": [-20, 20]},
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False
        },
        "action": {"type": "DiscreteMetaAction",
                    "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]},
        "lanes_count": env_config.get('lane_count', 2),
        "vehicles_count": env_config.get('vehicles_count', 45),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 45),
        "ego_spacing": 2,
        "collision_reward": -1.5,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.35,
        "lane_change_reward": -0.05,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False,
    }
    try:
        env = gym.make('roundabout-v1', render_mode=render_mode)
        env.unwrapped.configure(roundabout_config)
        logging.info("Created roundabout-v1 environment")
    except Exception as e:
        logging.warning(f"roundabout-v1 not available ({e}), using roundabout-v0")
        try:
            env = gym.make('roundabout-v0', render_mode=render_mode)
            env.unwrapped.configure(roundabout_config)
            logging.info("Created roundabout-v0 environment")
        except Exception as e2:
            logging.warning(f"roundabout-v0 not available ({e2}), using modified highway-v0")
            env = gym.make('highway-v0', render_mode=render_mode)
            env.unwrapped.configure(roundabout_config)
    return env


def create_racetrack_scenario(config, render_mode='rgb_array'):
    """
    Create Racetrack Scenario: High-speed overtaking coordination
    """
    env_config = config.get('environment', {}).get('racetrack', {})
    racetrack_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {"x": [-200, 200], "y": [-100, 100], "vx": [-40, 40], "vy": [-20, 20]},
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False
        },
        "action": {"type": "DiscreteMetaAction",
                    "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"]},
        "lanes_count": env_config.get('lane_count', 3),
        "vehicles_count": env_config.get('vehicles_count', 25),
        "controlled_vehicles": 1,
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 60),
        "ego_spacing": 3,
        "collision_reward": -3,
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.6,
        "lane_change_reward": 0.1,
        "reward_speed_range": [25, 40],
        "normalize_reward": True,
        "offroad_terminal": False,
    }
    try:
        env = gym.make('racetrack-v1', render_mode=render_mode)
        env.unwrapped.configure(racetrack_config)
        logging.info("Created racetrack-v1 environment")
    except Exception as e:
        logging.warning(f"racetrack-v1 not available ({e}), using racetrack-v0")
        try:
            env = gym.make('racetrack-v0', render_mode=render_mode)
            env.unwrapped.configure(racetrack_config)
            logging.info("Created racetrack-v0 environment")
        except Exception as e2:
            logging.warning(f"racetrack-v0 not available ({e2}), using modified highway-v0")
            env = gym.make('highway-v0', render_mode=render_mode)
            env.unwrapped.configure(racetrack_config)
    return env


def create_parking_lot_scenario(config, render_mode='rgb_array'):
    """
    Create Parking Lot Scenario: Complex maneuvering and space coordination
    
    A parking lot environment with multiple lanes, parking spaces, and complex
    maneuvering requirements. Tests parking assistance, space optimization,
    and courteous maneuvering behaviors.
    
    Args:
        config (dict): Configuration dictionary
        render_mode (str): Rendering mode ('human' for display, 'rgb_array' for headless)
        
    Returns:
        gym.Env: Configured parking lot environment
    """
    # Get environment parameters from config
    env_config = config.get('environment', {}).get('parking_lot', {})
    
    # Create parking lot environment with config
    parking_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 20,  # Increased to observe more vehicles
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "features_range": {
                "x": [-100, 100],
                "y": [-80, 80],
                "vx": [-10, 10],  # Lower max speed for parking lot
                "vy": [-10, 10],
                "heading": [-np.pi, np.pi]
            },
            "absolute": True,
            "order": "sorted",
            "see_ego": True,
            "normalize": False  # Use physical units so policy thresholds apply
        },
        "action": {
            "type": "DiscreteMetaAction",
            "actions": ["IDLE", "LANE_LEFT", "LANE_RIGHT", "FASTER", "SLOWER"],
        },
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": env_config.get('lanes_count', 4),  # Reduced for better parking lot layout
        "vehicles_count": env_config.get('vehicles_count', 15),  # Fewer total vehicles
        "controlled_vehicles": env_config.get('controlled_vehicles', 3),  # Multiple controlled vehicles
        "initial_lane_id": env_config.get('initial_lane_id', None),
        "duration": env_config.get('duration', 200),  # Longer episodes for parking
        "collision_reward": -5,  # Higher penalty for crashes
        "parking_reward": 2.0,  # Higher reward for successful parking
        "efficiency_reward": 0.5,
        "maneuvering_reward": 0.3,
        "reward_speed_range": [2, 8],  # Lower speeds for parking lot
        "normalize_reward": True,
        "offroad_terminal": True,  # Terminate if going off-road
        "offroad_reward": -3,  # Penalty for going off-road
        # Parking-specific parameters
        "parking_spaces": env_config.get('parking_spaces', 12),  # Fewer spaces for competition
        "maneuvering_zone_size": env_config.get('maneuvering_zone_size', 30),
        "parking_space_size": 4.0,  # Size of parking spaces
        "parking_tolerance": 1.0,  # Tolerance for parking success
        # Safety parameters
        "safe_distance": 2.0,  # Minimum safe distance
        "max_speed": 8.0,  # Maximum speed in parking lot
        "acceleration_range": [-3, 2],  # Conservative acceleration
    }
    
    # Create environment and configure it (use parking-v0)
    try:
        env = gym.make('parking-v0', render_mode=render_mode)
        env.unwrapped.configure(parking_config)
        logging.info("Created parking-v0 environment")
    except Exception as e:
        logging.warning(f"parking-v0 not available ({e}), trying parking-v1")
        try:
            env = gym.make('parking-v1', render_mode=render_mode)
            env.unwrapped.configure(parking_config)
            logging.info("Created parking-v1 environment")
        except Exception as e2:
            logging.error(f"Neither parking-v0 nor parking-v1 available: {e2}")
            raise RuntimeError("Parking environment not available. Please ensure highway-env is properly installed.")
    return env


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
        ("Intersection_100_Selfish", create_intersection_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Intersection_100_Cooperative", create_intersection_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Intersection_50_50_Mix", create_intersection_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        ("Roundabout_100_Selfish", create_roundabout_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Roundabout_100_Cooperative", create_roundabout_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Roundabout_50_50_Mix", create_roundabout_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        ("Racetrack_100_Selfish", create_racetrack_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("Racetrack_100_Cooperative", create_racetrack_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("Racetrack_50_50_Mix", create_racetrack_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
        ("ParkingLot_100_Selfish", create_parking_lot_scenario, {"selfish_ratio": 1.0, "cooperative_ratio": 0.0}),
        ("ParkingLot_100_Cooperative", create_parking_lot_scenario, {"selfish_ratio": 0.0, "cooperative_ratio": 1.0}),
        ("ParkingLot_50_50_Mix", create_parking_lot_scenario, {"selfish_ratio": 0.5, "cooperative_ratio": 0.5}),
    ]

    return scenarios


