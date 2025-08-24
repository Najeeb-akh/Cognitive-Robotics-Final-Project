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


def calculate_deltas_vs_baseline(results_df, baseline_composition="100% Selfish, 0% Cooperative"):
    """
    Calculate percentage changes vs selfish baseline for each scenario.
    
    Args:
        results_df (pd.DataFrame): Aggregated results dataframe
        baseline_composition (str): Name of baseline composition
        
    Returns:
        pd.DataFrame: Results with delta columns added
    """
    results_with_deltas = results_df.copy()
    
    # Initialize delta columns
    delta_columns = ['avg_speed_delta', 'total_collisions_delta', 'acceleration_std_delta', 'merge_success_rate_delta']
    for col in delta_columns:
        results_with_deltas[col] = np.nan
    
    scenarios = results_df['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario]
        
        # Find baseline values
        baseline_data = scenario_data[scenario_data['composition'] == baseline_composition]
        if baseline_data.empty:
            continue
            
        # Prefer network-wide speed if present
        speed_col = 'avg_network_speed_mean' if 'avg_network_speed_mean' in baseline_data.columns else 'avg_speed_mean'
        baseline_speed = baseline_data[speed_col].iloc[0]
        baseline_collisions = baseline_data['total_collisions_mean'].iloc[0]
        baseline_accel_std = baseline_data['acceleration_std_mean'].iloc[0]
        baseline_merge_rate = baseline_data['merge_success_rate_mean'].iloc[0]
        
        # Calculate deltas for all compositions in this scenario
        for idx, row in scenario_data.iterrows():
            if row['composition'] == baseline_composition:
                # Baseline has 0% change
                results_with_deltas.loc[results_with_deltas.index == idx, 'avg_speed_delta'] = 0.0
                results_with_deltas.loc[results_with_deltas.index == idx, 'total_collisions_delta'] = 0.0
                results_with_deltas.loc[results_with_deltas.index == idx, 'acceleration_std_delta'] = 0.0
                results_with_deltas.loc[results_with_deltas.index == idx, 'merge_success_rate_delta'] = 0.0
            else:
                # Calculate percentage changes
                if baseline_speed != 0:
                    row_speed = row.get('avg_network_speed_mean', np.nan)
                    if np.isnan(row_speed):
                        row_speed = row.get('avg_speed_mean', 0)
                    speed_delta = ((row_speed - baseline_speed) / baseline_speed) * 100
                    results_with_deltas.loc[results_with_deltas.index == idx, 'avg_speed_delta'] = speed_delta
                
                if baseline_collisions != 0:
                    collision_delta = ((row['total_collisions_mean'] - baseline_collisions) / baseline_collisions) * 100
                    results_with_deltas.loc[results_with_deltas.index == idx, 'total_collisions_delta'] = collision_delta
                elif row['total_collisions_mean'] == 0:
                    # Special case: baseline has collisions, this has none
                    results_with_deltas.loc[results_with_deltas.index == idx, 'total_collisions_delta'] = -100.0
                
                if baseline_accel_std != 0:
                    accel_delta = ((row['acceleration_std_mean'] - baseline_accel_std) / baseline_accel_std) * 100
                    results_with_deltas.loc[results_with_deltas.index == idx, 'acceleration_std_delta'] = accel_delta
                
                if baseline_merge_rate != 0:
                    merge_delta = ((row['merge_success_rate_mean'] - baseline_merge_rate) / baseline_merge_rate) * 100
                    results_with_deltas.loc[results_with_deltas.index == idx, 'merge_success_rate_delta'] = merge_delta
    
    return results_with_deltas


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
    
    # Calculate deltas vs baseline
    results_with_deltas = calculate_deltas_vs_baseline(results_df)
    
    # Generate plots for each scenario
    scenarios = results_with_deltas['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = results_with_deltas[results_with_deltas['scenario'] == scenario].copy()
        
        if scenario_data.empty:
            continue
        
        # 1. Average Network Speed vs Agent Composition (primary flow metric)
        plot_path = plot_average_network_speed(scenario_data, scenario, output_dir)
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
    plot_path = plot_combined_efficiency_comparison(results_with_deltas, output_dir)
    if plot_path:
        plot_paths.append(plot_path)
    
    plot_path = plot_combined_safety_comparison(results_with_deltas, output_dir)
    if plot_path:
        plot_paths.append(plot_path)
    
    # Generate summary dashboard
    plot_path = plot_summary_dashboard(results_with_deltas, output_dir)
    if plot_path:
        plot_paths.append(plot_path)
    
    return plot_paths


def plot_average_network_speed(scenario_data, scenario, output_dir):
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
        avg_speeds = scenario_data['avg_network_speed_mean'] if 'avg_network_speed_mean' in scenario_data.columns else scenario_data['avg_speed_mean']
        speed_stds = scenario_data['avg_network_speed_std'] if 'avg_network_speed_std' in scenario_data.columns else scenario_data['avg_speed_std']
        
        # Create bar chart
        bars = plt.bar(compositions, avg_speeds, yerr=speed_stds, 
                      capsize=5, alpha=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # Customize plot
        plt.title(f'{scenario} Scenario: Average Network Speed by Agent Composition')
        plt.xlabel('Agent Composition')
        plt.ylabel('Average Network Speed (m/s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars with sample size and delta information
        for i, (bar, speed, std) in enumerate(zip(bars, avg_speeds, speed_stds)):
            height = bar.get_height()
            row = scenario_data.iloc[i]
            sample_size = int(row.get('sample_size', 1))
            delta = row.get('avg_speed_delta', None)
            
            # Main value label
            label = f'{speed:.1f}±{std:.1f}'
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    label, ha='center', va='bottom', fontweight='bold')
            
            # Sample size label
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 1.5,
                    f'N={sample_size}', ha='center', va='bottom', fontsize=9, alpha=0.7)
            
            # Delta label (if not baseline)
            if delta is not None and not np.isnan(delta) and delta != 0:
                delta_str = f'{delta:+.1f}%'
                color = 'green' if delta > 0 else 'red'
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        delta_str, ha='center', va='center', 
                        fontweight='bold', color=color, fontsize=10)
        
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
        
        # Add value labels on bars with sample size and delta information
        for i, (bar, collision, std) in enumerate(zip(bars, collisions, collision_stds)):
            height = bar.get_height()
            row = scenario_data.iloc[i]
            sample_size = int(row.get('sample_size', 1))
            delta = row.get('total_collisions_delta', None)
            
            # Main value label
            label = f'{collision:.1f}±{std:.1f}'
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    label, ha='center', va='bottom', fontweight='bold')
            
            # Sample size label
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
                    f'N={sample_size}', ha='center', va='bottom', fontsize=9, alpha=0.7)
            
            # Delta label (if not baseline)
            if delta is not None and not np.isnan(delta) and delta != 0:
                delta_str = f'{delta:+.1f}%'
                # For collisions, green means fewer collisions (negative delta)
                color = 'green' if delta < 0 else 'red'
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        delta_str, ha='center', va='center', 
                        fontweight='bold', color=color, fontsize=10)
        
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
        
        # Add value labels on bars with sample size and delta information
        for i, (bar, std, std_std) in enumerate(zip(bars, accel_stds, accel_std_stds)):
            height = bar.get_height()
            row = scenario_data.iloc[i]
            sample_size = int(row.get('sample_size', 1))
            delta = row.get('acceleration_std_delta', None)
            
            # Main value label
            label = f'{std:.2f}±{std_std:.2f}'
            plt.text(bar.get_x() + bar.get_width()/2., height + std_std + 0.02,
                    label, ha='center', va='bottom', fontweight='bold')
            
            # Sample size label
            plt.text(bar.get_x() + bar.get_width()/2., height + std_std + 0.05,
                    f'N={sample_size}', ha='center', va='bottom', fontsize=9, alpha=0.7)
            
            # Delta label (if not baseline)
            if delta is not None and not np.isnan(delta) and delta != 0:
                delta_str = f'{delta:+.1f}%'
                # For acceleration stability, lower std is better (negative delta is good)
                color = 'green' if delta < 0 else 'red'
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        delta_str, ha='center', va='center', 
                        fontweight='bold', color=color, fontsize=10)
        
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
        
        # Add value labels on bars with sample size and delta information
        for i, (bar, rate, std) in enumerate(zip(bars, success_rates, success_stds)):
            height = bar.get_height()
            row = scenario_data.iloc[i]
            sample_size = int(row.get('sample_size', 1))
            delta = row.get('merge_success_rate_delta', None)
            
            # Main value label
            label = f'{rate:.1f}±{std:.1f}%'
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                    label, ha='center', va='bottom', fontweight='bold')
            
            # Sample size label
            plt.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                    f'N={sample_size}', ha='center', va='bottom', fontsize=9, alpha=0.7)
            
            # Delta label (if not baseline)
            if delta is not None and not np.isnan(delta) and delta != 0:
                delta_str = f'{delta:+.1f}%'
                # For merge success rate, higher is better (positive delta is good)
                color = 'green' if delta > 0 else 'red'
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        delta_str, ha='center', va='center', 
                        fontweight='bold', color=color, fontsize=10)
        
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
        
        # Plot 1: Average (Network) Speed
        for i, scenario in enumerate(scenarios):
            scenario_data = results_df[results_df['scenario'] == scenario]
            speed_col = 'avg_network_speed_mean' if 'avg_network_speed_mean' in scenario_data.columns else 'avg_speed_mean'
            speeds = [scenario_data[scenario_data['composition'] == comp][speed_col].values[0] 
                     if len(scenario_data[scenario_data['composition'] == comp]) > 0 else 0 
                     for comp in compositions]
            
            ax1.bar(x + i*width, speeds, width, label=scenario, alpha=0.8)
        
        ax1.set_title('Average Network Speed Comparison')
        ax1.set_xlabel('Agent Composition')
        ax1.set_ylabel('Average Network Speed (m/s)')
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


def plot_summary_dashboard(results_df, output_dir):
    """
    Generate a comprehensive summary dashboard comparing key KPIs across scenarios.
    
    Args:
        results_df (pd.DataFrame): Complete results dataframe with deltas
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved plot file
    """
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        scenarios = results_df['scenario'].unique()
        compositions = results_df['composition'].unique()
        
        # Filter to non-baseline compositions for delta display
        non_baseline_compositions = [comp for comp in compositions if "100% Selfish" not in comp]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        # Plot 1: Average Speed Deltas
        for i, comp in enumerate(non_baseline_compositions):
            comp_data = results_df[results_df['composition'] == comp]
            deltas = []
            for scenario in scenarios:
                scenario_comp_data = comp_data[comp_data['scenario'] == scenario]
                if not scenario_comp_data.empty:
                    delta = scenario_comp_data['avg_speed_delta'].iloc[0]
                    deltas.append(delta if not np.isnan(delta) else 0)
                else:
                    deltas.append(0)
            
            bars = ax1.bar(x + i*width, deltas, width, label=comp, alpha=0.8)
            
            # Add delta value labels
            for bar, delta in zip(bars, deltas):
                if delta != 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (0.5 if bar.get_height() >= 0 else -0.5),
                            f'{delta:+.1f}%', ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                            fontweight='bold', fontsize=9)
        
        ax1.set_title('Average Speed Change vs Baseline')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Speed Change (%)')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(scenarios)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Collision Deltas
        for i, comp in enumerate(non_baseline_compositions):
            comp_data = results_df[results_df['composition'] == comp]
            deltas = []
            for scenario in scenarios:
                scenario_comp_data = comp_data[comp_data['scenario'] == scenario]
                if not scenario_comp_data.empty:
                    delta = scenario_comp_data['total_collisions_delta'].iloc[0]
                    deltas.append(delta if not np.isnan(delta) else 0)
                else:
                    deltas.append(0)
            
            bars = ax2.bar(x + i*width, deltas, width, label=comp, alpha=0.8)
            
            # Add delta value labels
            for bar, delta in zip(bars, deltas):
                if delta != 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (1 if bar.get_height() >= 0 else -1),
                            f'{delta:+.1f}%', ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                            fontweight='bold', fontsize=9)
        
        ax2.set_title('Collision Change vs Baseline')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Collision Change (%)')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(scenarios)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 3: Acceleration Stability Deltas
        for i, comp in enumerate(non_baseline_compositions):
            comp_data = results_df[results_df['composition'] == comp]
            deltas = []
            for scenario in scenarios:
                scenario_comp_data = comp_data[comp_data['scenario'] == scenario]
                if not scenario_comp_data.empty:
                    delta = scenario_comp_data['acceleration_std_delta'].iloc[0]
                    deltas.append(delta if not np.isnan(delta) else 0)
                else:
                    deltas.append(0)
            
            bars = ax3.bar(x + i*width, deltas, width, label=comp, alpha=0.8)
            
            # Add delta value labels
            for bar, delta in zip(bars, deltas):
                if delta != 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (1 if bar.get_height() >= 0 else -1),
                            f'{delta:+.1f}%', ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                            fontweight='bold', fontsize=9)
        
        ax3.set_title('Acceleration Stability Change vs Baseline')
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('Accel Std Change (%)')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(scenarios)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 4: Sample Sizes
        for i, comp in enumerate(compositions):
            comp_data = results_df[results_df['composition'] == comp]
            sample_sizes = []
            for scenario in scenarios:
                scenario_comp_data = comp_data[comp_data['scenario'] == scenario]
                if not scenario_comp_data.empty:
                    sample_size = scenario_comp_data['sample_size'].iloc[0]
                    sample_sizes.append(sample_size)
                else:
                    sample_sizes.append(0)
            
            bars = ax4.bar(x + i*width, sample_sizes, width, label=comp, alpha=0.8)
            
            # Add sample size labels
            for bar, n in zip(bars, sample_sizes):
                if n > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                            f'{int(n)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax4.set_title('Sample Sizes by Scenario and Composition')
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Number of Runs (N)')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(scenarios)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Social Laws Simulation - Summary Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = 'summary_dashboard.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"Error generating summary dashboard: {e}")
        return None