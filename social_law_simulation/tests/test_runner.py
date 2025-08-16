import os
import sys

import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from scenarios import create_highway_scenario, create_merge_scenario
from metrics import MetricsCollector
from main import run_single_simulation as run_single_base


def short_config():
    return {
        'simulation': {'duration_steps': 5, 'num_runs_per_config': 1},
        'visualization': {'fps': 60},
        'metrics': {},
        'environment': {},
    }


@pytest.mark.parametrize('scenario_func', [create_highway_scenario, create_merge_scenario])
def test_base_scenarios_run_without_errors(scenario_func):
    cfg = short_config()
    env = scenario_func(cfg, render_mode='rgb_array')
    mc = MetricsCollector(cfg)
    metrics = run_single_base(env, None, cfg, mc, render=False)
    assert isinstance(metrics, dict)
    for key in ['total_collisions', 'avg_speed', 'steps']:
        assert key in metrics
    env.close()


def test_highway_run():
    cfg = short_config()
    env = create_highway_scenario(cfg, render_mode='rgb_array')
    mc = MetricsCollector(cfg)
    metrics = run_single_base(env, None, cfg, mc, render=False)
    assert isinstance(metrics, dict)
    for key in ['total_collisions', 'avg_speed', 'steps']:
        assert key in metrics
    env.close()


def test_merge_run():
    cfg = short_config()
    env = create_merge_scenario(cfg, render_mode='rgb_array')
    mc = MetricsCollector(cfg)
    metrics = run_single_base(env, None, cfg, mc, render=False)
    assert isinstance(metrics, dict)
    for key in ['total_collisions', 'avg_speed', 'steps']:
        assert key in metrics
    env.close()


