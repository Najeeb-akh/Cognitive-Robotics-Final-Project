"""
Main Simulation Runner (DEPRECATED)

This module now only exposes importable runner functions. CLI and setup logic
have been removed. Use `run_simulation.py` as the single entry point.

Deprecated: Functions remain for backward compatibility and import stability.
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
from datetime import datetime
import warnings

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scenarios import get_scenario_configurations
from metrics import MetricsCollector, MetricsAggregator
from visualization import generate_comparison_plots
from policies.selfish_policy import SelfishPolicy
from policies.cooperative_policy import CooperativePolicy


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config.yaml file
        
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
        "src/main.py setup_logging is deprecated; use run_simulation.py entry point",
        DeprecationWarning,
        stacklevel=2,
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(project_root, logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )


def create_agent_policy(agent_composition, config):
    """
    Create agent policy based on composition ratio.
    
    Args:
        agent_composition (dict): Dict with 'selfish_ratio' and 'cooperative_ratio'
        config (dict): Configuration dictionary
        
    Returns:
        Policy instance (SelfishPolicy or CooperativePolicy)
    """
    warnings.warn(
        "src/main.py create_agent_policy is deprecated; use run_simulation.py entry point",
        DeprecationWarning,
        stacklevel=2,
    )
    selfish_ratio = agent_composition.get('selfish_ratio', 0.5)
    
    if selfish_ratio > 0.5:
        # Create selfish policy
        policy = SelfishPolicy(config=config)
        logging.info("Created SelfishPolicy for ego vehicle")
    else:
        # Create cooperative policy
        policy = CooperativePolicy(config=config)
        logging.info("Created CooperativePolicy for ego vehicle")
    
    return policy


def run_single_simulation(env, agent_policy, config, metrics_collector, render=False, 
                          stop_event=None, pause_event=None, progress_cb=None):
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
        "src/main.py run_single_simulation is deprecated; use run_simulation.py entry point",
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
        log_actions = bool(config.get('debug', {}).get('log_actions', False))
        log_actions_steps = int(config.get('debug', {}).get('log_actions_steps', 0))
        log_ego_every = int(config.get('debug', {}).get('log_ego_every', 0))
        for step in range(duration_steps):
            # Check for stop event
            if stop_event and stop_event.is_set():
                logging.info(f"Simulation stopped by user at step {step}")
                break
            
            # Check for pause event
            if pause_event and pause_event.is_set():
                import time
                while pause_event.is_set() and not (stop_event and stop_event.is_set()):
                    time.sleep(0.05)  # Small sleep while paused
                if stop_event and stop_event.is_set():
                    logging.info(f"Simulation stopped while paused at step {step}")
                    break
            
            # Get action from agent policy if available, otherwise use default
            if agent_policy:
                action = agent_policy.act(obs)
                # Early-step action logging for diagnostics
                if log_actions and step < log_actions_steps:
                    # Best-effort ego speed read
                    ego = getattr(env.unwrapped, 'vehicle', None)
                    ego_speed = getattr(ego, 'speed', None) if ego is not None else None
                    logging.info(f"[ACTIONS] step={step} action={action} ego_speed={ego_speed}")
            else:
                action = 1  # Default fallback action (IDLE)
            
            # Step the environment with agent's action
            obs, reward, terminated, truncated, info = env.step(action)

            # No incremental population overrides; handled centrally in runner per-run
            
            # Render the environment if requested
            if render:
                env.render()
                # Add delay based on configured FPS
                import time
                fps = config.get('visualization', {}).get('fps', 20)
                time.sleep(1.0 / fps)  # Dynamic delay based on FPS setting
            
            # Collect metrics for this step (pass info per plan)
            metrics_collector.collect_step_metrics(env, step, info)

            # Periodic ego summary logging
            if log_ego_every and step % log_ego_every == 0:
                ego = getattr(env.unwrapped, 'vehicle', None)
                if ego is not None:
                    logging.info(
                        f"[EGO] step={step} speed={getattr(ego, 'speed', None)} lane_index={getattr(ego, 'lane_index', None)} crashed={getattr(ego, 'crashed', False)}"
                    )
            
            # Emit progress updates periodically
            if progress_cb and step % 50 == 0:
                partial_metrics = {
                    'avg_speed_so_far': float(np.mean(metrics_collector.speed_history)) if hasattr(metrics_collector, 'speed_history') and metrics_collector.speed_history else 0.0,
                    'total_collisions_so_far': int(getattr(metrics_collector, 'collisions', 0))
                }
                progress_cb(step, duration_steps, partial_metrics)
            
            # Check if simulation should end early
            if terminated or truncated:
                logging.info(f"Simulation ended early at step {step} - terminated: {terminated}, truncated: {truncated}")
                if info:
                    logging.info(f"Info: {info}")
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


def run_scenario_composition(scenario_name, scenario_func, composition, config, render_args=None,
                           stop_event=None, pause_event=None, progress_cb=None):
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
            
            # Create agent policy
            agent_policy = create_agent_policy(composition, config)
            
            # Create metrics collector
            metrics_collector = MetricsCollector(config)
            
            # Display rendering instructions if this is the first run with rendering
            if render_enabled and run_idx == 0:
                print("\n" + "="*60)
                print("🚗 REAL-TIME SIMULATION VISUALIZATION ACTIVE")
                print("="*60)
                print("• Watch the vehicles interact in real-time!")
                print("• Blue vehicles: Default highway-env agents")
                print("• Red vehicle: Ego vehicle (controlled)")
                print("• Green: Target lane/merge areas")
                print("• Close the visualization window to stop simulation")
                print("• Press ESC or Q in the window to exit early")
                print("="*60)
                
            # Run simulation with rendering if enabled
            metrics = run_single_simulation(env, agent_policy, config, metrics_collector, render=render_enabled,
                                          stop_event=stop_event, pause_event=pause_event, progress_cb=progress_cb)
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