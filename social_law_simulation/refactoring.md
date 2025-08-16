This plan outlines the refactoring of the traffic simulation project. The primary goals are to remove the tkinter launcher GUI, introduce a flexible command-line interface for running experiments, and establish a foundation for automated testing to ensure long-term stability and scalability.

Phase 1: highway-env API Essentials
This section outlines the core interaction patterns with the highway-env library, which form the basis of this plan.

Environment Instantiation: Environments are created using the Gymnasium API. For visualization, render_mode must be set.
import gymnasium as gym
env = gym.make("highway-v0", render_mode="human")
Environment Configuration: Custom parameters are applied immediately after creation and before the first reset().
env.unwrapped.configure({...})
Rendering: To display the simulation, render_mode='human' is required during instantiation. The env.render() method may be needed within the simulation loop for some setups.
Step Loop: The standard Gymnasium step function returns five values. The info dictionary is a critical source for detailed, environment-specific statistics.
obs, reward, terminated, truncated, info = env.step(action)
Key metrics like 'crashed' can be reliably sourced from info.
Files to be Removed
The following files and directories related to the legacy GUI will be permanently deleted.

social_law_simulation/gui/ (the entire directory and all its contents)
social_law_simulation/gui_app.py
Rationale: The new command-line-driven workflow makes the entire tkinter GUI launcher obsolete. Removing it completely decouples the core simulation logic from any specific UI framework.

Files to be Modified
These existing files will be updated to support the new structure.

social_law_simulation/src/main.py and social_law_simulation/src/main_extended.py
Action: Refactor the core simulation logic into clean, importable functions. The existing run_single_simulation(...) is a good candidate to be the target of this refactoring.
Goal: These functions should be able to accept a pre-configured environment object and return a dictionary of the final, aggregated metrics upon completion. They should not contain any argument parsing or environment setup logic, which will be moved to the new entry-point script.
Specific Change: Modify the internal simulation loop to capture the info dictionary from env.step() and pass it to the metrics collector at each step.
social_law_simulation/src/metrics.py
Action: Enhance the MetricsCollector class to process the info dictionary from the environment.
Goal: To capture more reliable, environment-provided data.
Specific Change: Add a new method, collect_step_info(self, info), that parses relevant data (e.g., 'crashed', 'speed'). This method should be called within the main collect_step_metrics(...) method to enrich the data collected at each timestep.
social_law_simulation/requirements.txt
Action: Add a new dependency for automated testing.
Specific Change: Add pytest to the file. Also, verify that highway-env, gymnasium, and pygame are present and up-to-date to ensure rendering works correctly.
New Files to be Created
These new files will form the new interface and quality assurance layer for the project.

social_law_simulation/run_simulation.py
Purpose: This script will be the single, primary entry point for all simulations, replacing all previous launcher scripts and the GUI.
Implementation: It will use Python's argparse library to handle command-line arguments.
Arguments to Implement:
--scenario: (Required) A string specifying the name of the scenario to run (e.g., highway, merge, intersection).
--composition: (Required) A string for the agent composition (e.g., selfish, cooperative, mixed).
--render: A boolean flag (action='store_true') to enable visualization (render_mode='human').
--output-csv: An optional string specifying the path to the main results CSV file where the aggregated run metrics will be appended.
--output-steps-csv: An optional string specifying the path to a file where per-step data for a single run will be saved.
--config-overrides: An optional JSON-formatted string to override default parameters from the config.yaml file (e.g., '{"duration": 500}').
Logic: The script will parse these arguments, load the base configuration file, apply any overrides, create the correct scenario and agent composition, and then call the refactored runner function from main.py or main_extended.py to execute the simulation.
social_law_simulation/tests/test_runner.py
Purpose: A new test file to contain automated "smoke tests" that ensure the core simulation logic remains functional after refactoring and in the future.
Implementation: This will be a pytest-compatible file.
Test Cases:
Create a separate test function for each primary scenario (test_highway_run, test_merge_run, etc.).
Each test will call the core simulation logic for a very short duration (e.g., 5 steps) with rendering disabled.
The test will assert that the simulation completes without raising any errors.
The test will assert that the returned metrics dictionary contains the expected keys (e.g., total_collisions, avg_speed, steps), ensuring the data collection pipeline is working.
Data Collection Strategy
Step-level: The MetricsCollector will now capture data from both the environment state and the info dictionary from env.step(). This provides a more robust source for critical events like crashes.
Run-level: The calculate_final_metrics() method will produce the final aggregated dictionary for a run, summarizing its performance.
Persistence:
The main results (one row per run) will be appended to the CSV file specified by the --output-csv command-line argument.
Detailed, per-step data can be saved to a separate CSV for deeper analysis by using the --output-steps-csv argument.
Step-by-Step Refactoring Guide
Follow these steps in order to ensure a smooth transition.

Remove GUI: Delete the social_law_simulation/gui/ directory and social_law_simulation/gui_app.py. This is the first step to ensure no legacy code is accidentally used.
Extend Metrics: In src/metrics.py, implement the changes to the MetricsCollector to process the info dictionary.
Refactor Runners: In src/main.py and src/main_extended.py, isolate the core simulation logic into clean, callable functions. Ensure they are updated to pass the info dict to the metrics system.
Create Generic Entry Point: Create the new run_simulation.py script. Implement the argument parsing (argparse) and the logic to set up and launch the correct scenario based on the command-line inputs.
Add Automated Tests: Create the tests/ directory (if it doesn't exist) and the tests/test_runner.py file. Write the smoke tests as described above. Add pytest to requirements.txt.
Run Tests: From the project's root directory, run pytest. All tests must pass before proceeding. This verifies that the core logic is still sound.
Manual Verification: Run the new run_simulation.py script with the --render flag for a few different scenarios to visually confirm that the highway-env visualization window appears and animates as expected.
Update Documentation: Update the project's README.md to remove instructions about the old GUI. Add a section that points to the new usage_guide.md for instructions on how to run simulations with the new script.