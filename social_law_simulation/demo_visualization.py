#!/usr/bin/env python3
"""
Demo Script for Real-Time Visualization

This script provides easy commands to demonstrate the simulation with real-time visualization.
Run this to see the different agent behaviors in action!
"""

import subprocess
import sys
import os

def run_demo(scenario, composition, description):
    """Run a demo simulation with visualization."""
    print(f"\n{'='*60}")
    print(f"🚗 DEMO: {description}")
    print(f"{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"Composition: {composition}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "src/main.py", 
        "--config", "test_config.yaml",
        "--render", 
        "--scenario", scenario,
        "--composition", composition,
        "--log-level", "INFO"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Demo failed: {e}")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")

def main():
    """Main demo function."""
    print("🚗 HIGHWAY TRAFFIC SIMULATION - VISUALIZATION DEMOS")
    print("="*60)
    
    demos = [
        ("Highway", "100% Selfish", "Standard selfish driving behavior"),
        ("Highway", "100% Cooperative", "Cooperative driving with social laws"),  
        ("Merge", "100% Selfish", "Selfish behavior at merge points"),
        ("Merge", "100% Cooperative", "Cooperative merging and yielding"),
    ]
    
    while True:
        print("\nAvailable demos:")
        for i, (scenario, composition, desc) in enumerate(demos, 1):
            print(f"{i}. {desc}")
        print("5. Exit")
        
        try:
            choice = input("\nSelect demo (1-5): ").strip()
            
            if choice == "5":
                print("Goodbye!")
                break
            elif choice in ["1", "2", "3", "4"]:
                idx = int(choice) - 1
                scenario, composition, description = demos[idx]
                run_demo(scenario, composition, description)
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("src/main.py"):
        print("Error: Please run this script from the social_law_simulation directory")
        print("Usage: cd social_law_simulation && python demo_visualization.py")
        sys.exit(1)
    
    main()