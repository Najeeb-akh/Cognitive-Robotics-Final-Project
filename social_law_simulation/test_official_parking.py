#!/usr/bin/env python3
"""
Test script for the Official Parking Policy

This script demonstrates how to use the new official parking policy
based on the highway-env parking environment approach.

Usage:
    python test_official_parking.py
"""

import sys
import os
import yaml
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.policy_factory import create_official_parking_policy
from src.scenarios import create_parking_lot_scenario

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found")
        return None

def test_official_parking_policy():
    """Test the official parking policy"""
    print("Testing Official Parking Policy")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    if config is None:
        return
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create parking lot environment
    print("Creating parking lot environment...")
    env = create_parking_lot_scenario(config, render_mode='rgb_array')
    
    # Create official parking policy
    print("Creating official parking policy...")
    agent_composition = {
        'selfish_ratio': 0.0,
        'cooperative_ratio': 1.0,
        'defensive_ratio': 0.0
    }
    
    policy = create_official_parking_policy(agent_composition, config)
    print(f"Created policy: {policy.__class__.__name__}")
    
    # Test the policy with a few steps
    print("\nTesting policy with environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
    print(f"Initial observation length: {len(obs)}")
    
    # Run a few steps
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        action = policy.act(obs)
        print(f"Action taken: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode ended early")
            break
    
    print("\nTest completed successfully!")

def compare_policies():
    """Compare the original and official parking policies"""
    print("\nComparing Parking Policies")
    print("=" * 50)
    
    config = load_config()
    if config is None:
        return
    
    # Create both policies
    agent_composition = {
        'selfish_ratio': 0.0,
        'cooperative_ratio': 1.0,
        'defensive_ratio': 0.0
    }
    
    # Original policy
    from src.policy_factory import create_agent_policy
    original_policy = create_agent_policy(agent_composition, config, 'parking_lot')
    
    # Official policy
    official_policy = create_official_parking_policy(agent_composition, config)
    
    print(f"Original policy: {original_policy.__class__.__name__}")
    print(f"Official policy: {official_policy.__class__.__name__}")
    
    # Test both with same observation
    env = create_parking_lot_scenario(config, render_mode='rgb_array')
    obs, info = env.reset()
    
    print(f"\nTesting with observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
    
    # Test original policy
    print("\n--- Original Policy ---")
    original_action = original_policy.act(obs)
    print(f"Original action: {original_action}")
    
    # Test official policy
    print("\n--- Official Policy ---")
    official_action = official_policy.act(obs)
    print(f"Official action: {official_action}")
    
    print("\nComparison completed!")

if __name__ == "__main__":
    print("Official Parking Policy Test")
    print("Based on: https://highway-env.farama.org/environments/parking/")
    print("=" * 60)
    
    try:
        test_official_parking_policy()
        compare_policies()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
