"""
Visualization and Reporting Implementation (FR6)

This module generates comparison plots and visualizations for the simulation results.
Implements FR6.2 requirements for bar charts comparing different agent compositions.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def setup_plot_style():
    """Setup consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def generate_comparison_plots(results_df, output_dir):
    """
    Generate all comparison plots as specified in FR6.2.
    
    Args:
        results_df (pd.DataFrame): Aggregated results dataframe
        output_dir (str): Directory to save plots
        
    Returns:
        list: List of saved plot file paths
    """
    setup_plot_style()
    
    if results_df.empty:
        print("No data available for plotting")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []
    
    # Generate plots for each scenario
    scenarios = results_df['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario].copy()
        
        if scenario_data.empty:
            continue
        
        # 1. Average Speed vs Agent Composition
        plot_path = plot_average_speed(scenario_data, scenario, output_dir)
        if plot_path:
            plot_paths.append(plot_path)
        
        # 2. Total Collisions vs Agent Composition  
        plot_path = plot_collisions(scenario_data, scenario, output_dir)
        if plot_path:
            plot_paths.append(plot_path)
        
        # 3. Standard Deviation of Acceleration vs Agent Composition
        plot_path = plot_acceleration_stability(scenario_data, scenario, output_dir)
        if plot_path:
            plot_paths.append(plot_path)
        
        # 4. Merge Success Rate (for merge scenarios only)
        if scenario.lower() == 'merge':
            plot_path = plot_merge_success_rate(scenario_data, scenario, output_dir)
            if plot_path:
                plot_paths.append(plot_path)
    
    # Generate combined comparison plots
    plot_path = plot_combined_efficiency_comparison(results_df, output_dir)
    if plot_path:
        plot_paths.append(plot_path)
    
    plot_path = plot_combined_safety_comparison(results_df, output_dir)
    if plot_path:
        plot_paths.append(plot_path)
    
    return plot_paths


def plot_average_speed(scenario_data, scenario, output_dir):
    """
    Generate bar chart for Average Speed vs Agent Composition.
    
    Args:
        scenario_data (pd.DataFrame): Data for specific scenario
        scenario (str): Scenario name
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        compositions = scenario_data['composition']
        avg_speeds = scenario_data['avg_speed_mean']
        speed_stds = scenario_data['avg_speed_std']
        
        # Create bar chart
        bars = plt.bar(compositions, avg_speeds, yerr=speed_stds, 
                      capsize=5, alpha=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # Customize plot
        plt.title(f'{scenario} Scenario: Average Speed by Agent Composition')
        plt.xlabel('Agent Composition')
        plt.ylabel('Average Speed (m/s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, speed, std in zip(bars, avg_speeds, speed_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{speed:.1f}±{std:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{scenario.lower()}_average_speed.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating average speed plot for {scenario}: {e}")
        return None


def plot_collisions(scenario_data, scenario, output_dir):
    """
    Generate bar chart for Total Collisions vs Agent Composition.
    
    Args:
        scenario_data (pd.DataFrame): Data for specific scenario
        scenario (str): Scenario name
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        compositions = scenario_data['composition']
        collisions = scenario_data['total_collisions_mean']
        collision_stds = scenario_data['total_collisions_std']
        
        # Create bar chart
        bars = plt.bar(compositions, collisions, yerr=collision_stds,
                      capsize=5, alpha=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # Customize plot
        plt.title(f'{scenario} Scenario: Total Collisions by Agent Composition')
        plt.xlabel('Agent Composition')
        plt.ylabel('Total Collisions')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, collision, std in zip(bars, collisions, collision_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{collision:.1f}±{std:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{scenario.lower()}_collisions.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating collisions plot for {scenario}: {e}")
        return None


def plot_acceleration_stability(scenario_data, scenario, output_dir):
    """
    Generate bar chart for Acceleration Standard Deviation vs Agent Composition.
    
    Args:
        scenario_data (pd.DataFrame): Data for specific scenario
        scenario (str): Scenario name
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        compositions = scenario_data['composition']
        accel_stds = scenario_data['acceleration_std_mean']
        accel_std_stds = scenario_data['acceleration_std_std']
        
        # Create bar chart
        bars = plt.bar(compositions, accel_stds, yerr=accel_std_stds,
                      capsize=5, alpha=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # Customize plot
        plt.title(f'{scenario} Scenario: Acceleration Stability by Agent Composition')
        plt.xlabel('Agent Composition')
        plt.ylabel('Acceleration Std Dev (m/s²)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, std, std_std in zip(bars, accel_stds, accel_std_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std_std,
                    f'{std:.2f}±{std_std:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{scenario.lower()}_acceleration_stability.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating acceleration stability plot for {scenario}: {e}")
        return None


def plot_merge_success_rate(scenario_data, scenario, output_dir):
    """
    Generate bar chart for Merge Success Rate vs Agent Composition.
    
    Args:
        scenario_data (pd.DataFrame): Data for specific scenario
        scenario (str): Scenario name
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract data
        compositions = scenario_data['composition']
        success_rates = scenario_data['merge_success_rate_mean'] * 100  # Convert to percentage
        success_stds = scenario_data['merge_success_rate_std'] * 100
        
        # Create bar chart
        bars = plt.bar(compositions, success_rates, yerr=success_stds,
                      capsize=5, alpha=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # Customize plot
        plt.title(f'{scenario} Scenario: Merge Success Rate by Agent Composition')
        plt.xlabel('Agent Composition')
        plt.ylabel('Merge Success Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate, std in zip(bars, success_rates, success_stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{rate:.1f}±{std:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'{scenario.lower()}_merge_success_rate.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating merge success rate plot for {scenario}: {e}")
        return None


def plot_combined_efficiency_comparison(results_df, output_dir):
    """
    Generate combined plot comparing efficiency across scenarios.
    
    Args:
        results_df (pd.DataFrame): Complete results dataframe
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scenarios = results_df['scenario'].unique()
        compositions = results_df['composition'].unique()
        
        x = np.arange(len(compositions))
        width = 0.35
        
        # Plot 1: Average Speed
        for i, scenario in enumerate(scenarios):
            scenario_data = results_df[results_df['scenario'] == scenario]
            speeds = [scenario_data[scenario_data['composition'] == comp]['avg_speed_mean'].values[0] 
                     if len(scenario_data[scenario_data['composition'] == comp]) > 0 else 0 
                     for comp in compositions]
            
            ax1.bar(x + i*width, speeds, width, label=scenario, alpha=0.8)
        
        ax1.set_title('Average Speed Comparison')
        ax1.set_xlabel('Agent Composition')
        ax1.set_ylabel('Average Speed (m/s)')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(compositions, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Throughput (if available)
        for i, scenario in enumerate(scenarios):
            scenario_data = results_df[results_df['scenario'] == scenario]
            throughputs = [scenario_data[scenario_data['composition'] == comp]['throughput_mean'].values[0]
                          if len(scenario_data[scenario_data['composition'] == comp]) > 0 else 0
                          for comp in compositions]
            
            ax2.bar(x + i*width, throughputs, width, label=scenario, alpha=0.8)
        
        ax2.set_title('Throughput Comparison')
        ax2.set_xlabel('Agent Composition')
        ax2.set_ylabel('Vehicles Completed')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(compositions, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = 'combined_efficiency_comparison.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating combined efficiency plot: {e}")
        return None


def plot_combined_safety_comparison(results_df, output_dir):
    """
    Generate combined plot comparing safety across scenarios.
    
    Args:
        results_df (pd.DataFrame): Complete results dataframe
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scenarios = results_df['scenario'].unique()
        compositions = results_df['composition'].unique()
        
        x = np.arange(len(compositions))
        width = 0.35
        
        # Plot 1: Collisions
        for i, scenario in enumerate(scenarios):
            scenario_data = results_df[results_df['scenario'] == scenario]
            collisions = [scenario_data[scenario_data['composition'] == comp]['total_collisions_mean'].values[0]
                         if len(scenario_data[scenario_data['composition'] == comp]) > 0 else 0
                         for comp in compositions]
            
            ax1.bar(x + i*width, collisions, width, label=scenario, alpha=0.8)
        
        ax1.set_title('Collision Comparison')
        ax1.set_xlabel('Agent Composition')
        ax1.set_ylabel('Total Collisions')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(compositions, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: TTC Events
        for i, scenario in enumerate(scenarios):
            scenario_data = results_df[results_df['scenario'] == scenario]
            ttc_events = [scenario_data[scenario_data['composition'] == comp]['ttc_events_count_mean'].values[0]
                         if len(scenario_data[scenario_data['composition'] == comp]) > 0 else 0
                         for comp in compositions]
            
            ax2.bar(x + i*width, ttc_events, width, label=scenario, alpha=0.8)
        
        ax2.set_title('Near-Miss Events (TTC < 2s)')
        ax2.set_xlabel('Agent Composition')
        ax2.set_ylabel('TTC Events Count')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(compositions, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = 'combined_safety_comparison.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating combined safety plot: {e}")
        return None