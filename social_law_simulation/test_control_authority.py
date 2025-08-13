"""
Control Authority Test

This test verifies that policies can actually control vehicles.
It creates a minimal test with a single agent that has a hardcoded policy
to constantly command lane changes, and verifies the vehicle responds.

Test Success Criteria:
- Vehicle must visibly and verifiably change lanes in rendered simulation
- Agent's actions must directly control vehicle behavior
"""

import gymnasium as gym
import highway_env
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scenarios import create_highway_scenario
import yaml

class ConstantLaneChangeAgent:
    """
    Test agent that constantly tries to change lanes.
    This is designed to test if our custom agents can control vehicles.
    """
    def __init__(self):
        self.step_count = 0
        self.target_lane = None
        self.last_action = 1  # Start with IDLE
        
    def act(self):
        """
        Hardcoded policy: alternate between left and right lane changes
        every 20 steps to create visible, predictable behavior.
        """
        self.step_count += 1
        
        # Change direction every 20 steps
        if self.step_count % 20 == 0:
            if self.last_action != 0:  # If not going left
                self.last_action = 0  # LANE_LEFT
            else:
                self.last_action = 2  # LANE_RIGHT
        
        return self.last_action


def test_control_authority():
    """
    Main test function for control authority.
    """
    print("="*60)
    print("CONTROL AUTHORITY TEST")
    print("="*60)
    print("Testing if custom agents can actually control vehicles...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment with human rendering to visually verify
    env = create_highway_scenario(config, render_mode='human')
    
    # Create test agent
    test_agent = ConstantLaneChangeAgent()
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Initial observation shape: {np.array(obs).shape}")
    print(f"Environment action space: {env.action_space}")
    print(f"Environment vehicles: {len(env.unwrapped.road.vehicles)} vehicles")
    
    print("\nStarting test simulation...")
    print("Watch for lane changes every ~20 steps")
    print("Close the window or press Ctrl+C to stop the test")
    
    successful_actions = 0
    lane_changes_detected = 0
    previous_lane = None
    
    try:
        for step in range(200):  # Run for 200 steps
            # Get action from our test agent
            action = test_agent.act()
            
            # Get current ego vehicle lane before action
            ego_vehicle = env.unwrapped.road.vehicles[0]  # Ego vehicle is first
            current_lane = ego_vehicle.lane_index[2] if hasattr(ego_vehicle, 'lane_index') else None
            
            # Track lane changes
            if previous_lane is not None and current_lane != previous_lane:
                lane_changes_detected += 1
                print(f"Step {step}: LANE CHANGE DETECTED! {previous_lane} -> {current_lane}")
            
            # Apply action to environment - this should now work with our fixed main.py
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track successful actions (non-idle)
            if action != 1:
                successful_actions += 1
            
            # Render
            env.render()
            
            # Log progress
            if step % 50 == 0:
                print(f"Step {step}: Action={action}, Lane={current_lane}, Changes={lane_changes_detected}")
            
            previous_lane = current_lane
            
            # Check for early termination
            if terminated or truncated:
                print(f"Simulation terminated early at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    finally:
        env.close()
    
    # Report results
    print("\n" + "="*60)
    print("CONTROL AUTHORITY TEST RESULTS")
    print("="*60)
    print(f"Total steps executed: {step + 1}")
    print(f"Non-idle actions commanded: {successful_actions}")
    print(f"Lane changes detected: {lane_changes_detected}")
    
    # Determine test result
    if lane_changes_detected > 0:
        print("✅ CONTROL AUTHORITY TEST: PASS")
        print("   - Agent successfully controlled vehicle behavior")
        print(f"   - Detected {lane_changes_detected} lane changes")
        return True
    else:
        print("❌ CONTROL AUTHORITY TEST: FAIL")
        print("   - No lane changes detected despite commanding lane change actions")
        print("   - Vehicle may not be responding to agent commands")
        return False


if __name__ == "__main__":
    success = test_control_authority()
    sys.exit(0 if success else 1)