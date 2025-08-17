"""
Extended Main Simulation Runner (DEPRECATED)

This module now only exposes importable runner functions for compatibility.
Use `run_simulation.py` as the single entry point.
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
import warnings

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import extended components
from scenarios_extended import (get_extended_scenario_configurations, 
                               get_original_scenarios_only, 
                               get_new_scenarios_only)
from metrics import MetricsCollector, MetricsAggregator
from visualization import generate_comparison_plots

# Import original and extended policies
from policies.selfish_policy import SelfishPolicy
from policies.cooperative_policy import CooperativePolicy
from policies.intersection_policy import IntersectionCooperativePolicy, IntersectionSelfishPolicy
from policies.roundabout_policy import RoundaboutCooperativePolicy, RoundaboutSelfishPolicy
from policies.racetrack_policy import RacetrackCooperativePolicy, RacetrackSelfishPolicy


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        raise


def setup_logging(log_level='INFO', logs_dir='logs'):
    """Setup logging configuration (deprecated module)."""
    warnings.warn(
        "src/main_extended.py setup_logging is deprecated; use run_simulation.py entry point",
        DeprecationWarning,
        stacklevel=2,
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(project_root, logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'simulation_extended_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )


def detect_scenario_type(scenario_name):
    """
    Detect scenario type from scenario name.
    
    Args:
        scenario_name (str): Name of the scenario
        
    Returns:
        str: Scenario type ('highway', 'merge', 'intersection', 'roundabout', 'racetrack')
    """
    scenario_name_lower = scenario_name.lower()
    
    if 'intersection' in scenario_name_lower:
        return 'intersection'
    elif 'roundabout' in scenario_name_lower:
        return 'roundabout'
    elif 'racetrack' in scenario_name_lower:
        return 'racetrack'
    elif 'merge' in scenario_name_lower:
        return 'merge'
    elif 'highway' in scenario_name_lower:
        return 'highway'
    else:
        return 'highway'  # Default fallback


def create_agent_policy(agent_composition, config, scenario_type='highway'):
    """
    Create agent policy based on composition ratio and scenario type.
    
    Args:
        agent_composition (dict): Dict with 'selfish_ratio' and 'cooperative_ratio'
        config (dict): Configuration dictionary
        scenario_type (str): Type of scenario for specialized policies
        
    Returns:
        Policy instance
    """
    warnings.warn(
        "src/main_extended.py create_agent_policy is deprecated; use run_simulation.py entry point",
        DeprecationWarning,
        stacklevel=2,
    )
    selfish_ratio = agent_composition.get('selfish_ratio', 0.5)
    
    if selfish_ratio > 0.5:
        # Create scenario-specific selfish policy
        if scenario_type == 'intersection':
            policy = IntersectionSelfishPolicy(config=config)
            logging.info("Created IntersectionSelfishPolicy for ego vehicle")
        elif scenario_type == 'roundabout':
            policy = RoundaboutSelfishPolicy(config=config)
            logging.info("Created RoundaboutSelfishPolicy for ego vehicle")
        elif scenario_type == 'racetrack':
            policy = RacetrackSelfishPolicy(config=config)
            logging.info("Created RacetrackSelfishPolicy for ego vehicle")
        else:
            # Original scenarios use original policy
            policy = SelfishPolicy(config=config)
            logging.info("Created SelfishPolicy for ego vehicle")
    else:
        # Create scenario-specific cooperative policy
        if scenario_type == 'intersection':
            policy = IntersectionCooperativePolicy(config=config)
            logging.info("Created IntersectionCooperativePolicy for ego vehicle")
        elif scenario_type == 'roundabout':
            policy = RoundaboutCooperativePolicy(config=config)
            logging.info("Created RoundaboutCooperativePolicy for ego vehicle")
        elif scenario_type == 'racetrack':
            policy = RacetrackCooperativePolicy(config=config)
            logging.info("Created RacetrackCooperativePolicy for ego vehicle")
        else:
            # Original scenarios use original policy
            policy = CooperativePolicy(config=config)
            logging.info("Created CooperativePolicy for ego vehicle")
    
    return policy


def run_single_simulation(env, agent_policy, config, metrics_collector, render=False):
    """
    Run a single simulation with specified environment and agent policy.
    
    Args:
        env: Configured highway environment
        agent_policy: Agent policy function that takes obs and returns action
        config (dict): Configuration dictionary
        metrics_collector: MetricsCollector instance
        render (bool): Whether to render the simulation in real-time
        
    Returns:
        dict: Metrics from the simulation run
    """
    warnings.warn(
        "src/main_extended.py run_single_simulation is deprecated; use run_simulation.py entry point",
        DeprecationWarning,
        stacklevel=2,
    )
    duration_steps = config.get('simulation', {}).get('duration_steps', 1000)
    
    # Reset environment and metrics
    obs, info = env.reset()
    metrics_collector.reset_metrics()
    
    agent_name = agent_policy.__class__.__name__ if agent_policy else "DefaultPolicy"
    logging.info(f"Starting simulation with {agent_name} for {duration_steps} steps")
    
    try:
        for step in range(duration_steps):
            # Get action from agent policy if available, otherwise use default
            if agent_policy:
                action = agent_policy.act(obs)
            else:
                action = 1  # Default fallback action (IDLE)
            
            # Step the environment with agent's action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment if requested
            if render:
                env.render()
                # Add delay based on configured FPS
                import time
                fps = config.get('visualization', {}).get('fps', 20)
                time.sleep(1.0 / fps)  # Dynamic delay based on FPS setting
            
            # Collect metrics for this step with info
            metrics_collector.collect_step_metrics(env, step, info)
            
            # Check if simulation should end early
            if terminated or truncated:
                logging.info(f"Simulation ended early at step {step}")
                break
            
            # Log progress periodically
            if step % 200 == 0:
                logging.info(f"Step {step}/{duration_steps} completed")
    
    except Exception as e:
        logging.error(f"Error during simulation: {e}")
        raise
    
    # Calculate final metrics
    final_metrics = metrics_collector.calculate_final_metrics()
    logging.info("Simulation completed successfully")
    
    return final_metrics


def run_scenario_composition(scenario_name, scenario_func, composition, config, render_args=None):
    """
    Run multiple simulation runs for a specific scenario and agent composition.
    
    Args:
        scenario_name (str): Name of the scenario
        scenario_func: Function to create scenario environment
        composition (dict): Agent composition ratios
        config (dict): Configuration dictionary
        render_args (dict): Rendering arguments with 'enabled' and 'mode' keys
        
    Returns:
        list: List of metrics from all runs
    """
    num_runs = config.get('simulation', {}).get('num_runs_per_config', 5)
    composition_name = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
    
    logging.info(f"Running {scenario_name} with {composition_name} - {num_runs} runs")
    
    # Detect scenario type for specialized policies
    scenario_type = detect_scenario_type(scenario_name)
    
    run_metrics = []
    
    for run_idx in range(num_runs):
        logging.info(f"Starting run {run_idx + 1}/{num_runs}")
        
        try:
            # Determine render mode
            render_mode = 'rgb_array'  # Default
            render_enabled = False
            if render_args and render_args.get('enabled', False):
                render_mode = render_args.get('mode', 'human')
                render_enabled = True
            
            # Create environment with appropriate render mode
            env = scenario_func(config, render_mode=render_mode)
            
            # Create agent policy with scenario-specific specialization
            agent_policy = create_agent_policy(composition, config, scenario_type)
            
            # Create metrics collector
            metrics_collector = MetricsCollector(config)
            
            # Display rendering instructions if this is the first run with rendering
            if render_enabled and run_idx == 0:
                print("\n" + "="*80)
                print(f"🚗 EXTENDED SIMULATION VISUALIZATION - {scenario_name.upper()}")
                print("="*80)
                if scenario_type == 'intersection':
                    print("• Watch multi-directional traffic coordination!")
                    print("• Observe turn-taking and gap provision behaviors")
                elif scenario_type == 'roundabout':
                    print("• Watch circular traffic flow and entry coordination!")
                    print("• Observe entry facilitation and exit courtesy")
                elif scenario_type == 'racetrack':
                    print("• Watch high-speed racing and overtaking coordination!")
                    print("• Observe slipstream cooperation and defensive positioning")
                else:
                    print("• Watch the original scenario with enhanced social laws!")
                print("• Blue vehicles: Default highway-env agents")
                print("• Red vehicle: Ego vehicle (controlled)")
                print("• Green: Target areas/zones")
                print("• Close the visualization window to stop simulation")
                print("• Press ESC or Q in the window to exit early")
                print("="*80)
                
            # Run simulation with rendering if enabled
            metrics = run_single_simulation(env, agent_policy, config, metrics_collector, render=render_enabled)
            run_metrics.append(metrics)
            
            logging.info(f"Run {run_idx + 1} completed - Avg Speed: {metrics.get('avg_speed', 0):.2f}, Collisions: {metrics.get('total_collisions', 0)}")
            
            # Clean up
            env.close()
            
        except Exception as e:
            import traceback
            logging.error(f"Error in run {run_idx + 1}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    return run_metrics


"""
This module now only exposes importable runner functions. CLI and setup logic
have been removed per refactoring plan; use run_simulation.py for entry point.
"""