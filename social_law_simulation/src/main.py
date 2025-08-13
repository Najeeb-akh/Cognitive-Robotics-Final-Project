"""
Main Simulation Runner

This script orchestrates the entire simulation study according to the SRS requirements.
It runs all scenario/composition combinations, collects metrics, and generates reports.
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

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


def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
            
            # Collect metrics for this step
            metrics_collector.collect_step_metrics(env, step)
            
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Decentralized Highway Traffic Simulation')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--scenario', help='Run specific scenario only (e.g., "Highway", "Merge")')
    parser.add_argument('--composition', help='Run specific composition only (e.g., "100%% Selfish")')
    parser.add_argument('--render', action='store_true', help='Enable real-time visualization of the simulation')
    parser.add_argument('--render-mode', default='human', choices=['human', 'rgb_array'], 
                       help='Rendering mode: human for window display, rgb_array for programmatic access')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting Decentralized Highway Traffic Simulation")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logging.info(f"Configuration loaded from {args.config}")
        
        # Create output directories
        results_dir = config.get('output', {}).get('results_dir', 'results')
        plots_dir = config.get('output', {}).get('plots_dir', 'plots')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get scenario configurations
        scenario_configs = get_scenario_configurations()
        
        # Filter scenarios if specified
        if args.scenario:
            scenario_configs = [sc for sc in scenario_configs if args.scenario.lower() in sc[0].lower()]
        
        # Initialize metrics aggregator
        aggregator = MetricsAggregator()
        
        # Prepare render arguments
        render_args = {
            'enabled': args.render,
            'mode': args.render_mode
        }
        
        # Run all scenario/composition combinations
        total_configs = len(scenario_configs)
        for idx, (scenario_name, scenario_func, composition) in enumerate(scenario_configs):
            logging.info(f"Processing configuration {idx + 1}/{total_configs}: {scenario_name}")
            
            # Filter compositions if specified
            composition_desc = f"{int(composition['selfish_ratio']*100)}% Selfish, {int(composition['cooperative_ratio']*100)}% Cooperative"
            if args.composition and args.composition not in composition_desc:
                continue
            
            try:
                # Run multiple simulations for this configuration
                run_metrics = run_scenario_composition(scenario_name, scenario_func, composition, config, render_args)
                
                # Add to aggregator
                for metrics in run_metrics:
                    aggregator.add_run_metrics(
                        scenario_name.split('_')[0],  # Extract "Highway" or "Merge"
                        composition_desc,
                        metrics
                    )
                
                logging.info(f"Completed {scenario_name} with {len(run_metrics)} successful runs")
                
            except Exception as e:
                logging.error(f"Failed to complete {scenario_name}: {e}")
                continue
        
        # Generate results and reports
        logging.info("Generating final results and visualizations")
        
        # Save aggregated results (FR6.1)
        results_filename = config.get('output', {}).get('data_filename', 'results.csv')
        results_path = aggregator.save_results(results_dir, results_filename)
        logging.info(f"Results saved to {results_path}")
        
        # Generate comparison plots (FR6.2) 
        try:
            plot_paths = generate_comparison_plots(aggregator.get_aggregated_results(), plots_dir)
            logging.info(f"Plots saved to {plots_dir}")
            for plot_path in plot_paths:
                logging.info(f"  - {plot_path}")
        except Exception as e:
            logging.error(f"Failed to generate plots: {e}")
        
        # Print summary
        results_df = aggregator.get_aggregated_results()
        if not results_df.empty:
            print("\n" + "="*80)
            print("SIMULATION RESULTS SUMMARY")
            print("="*80)
            print(results_df.to_string(index=False))
            print("="*80)
        
        logging.info("Simulation study completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()