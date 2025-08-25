from policies.selfish_policy import SelfishPolicy
from policies.cooperative_policy import CooperativePolicy
from policies.intersection_policy import IntersectionCooperativePolicy, IntersectionSelfishPolicy
from policies.roundabout_policy import RoundaboutCooperativePolicy, RoundaboutSelfishPolicy
from policies.racetrack_policy import RacetrackCooperativePolicy, RacetrackSelfishPolicy


def detect_scenario_type(scenario_name: str) -> str:
    name = (scenario_name or '').lower()
    if 'intersection' in name:
        return 'intersection'
    if 'roundabout' in name:
        return 'roundabout'
    if 'racetrack' in name:
        return 'racetrack'
    if 'merge' in name:
        return 'merge'
    if 'highway' in name:
        return 'highway'
    return 'highway'


def create_agent_policy(agent_composition: dict, config: dict, scenario_type: str | None = None):
    import random as _random
    selfish_ratio = float(agent_composition.get('selfish_ratio', 0.5))
    # Randomly select ego type according to selfish_ratio (seeded upstream)
    is_selfish = bool(_random.random() < selfish_ratio)
    st = (scenario_type or '').strip().lower()

    if st == 'intersection':
        return (IntersectionSelfishPolicy(config) if is_selfish
                else IntersectionCooperativePolicy(config))
    if st == 'roundabout':
        return (RoundaboutSelfishPolicy(config) if is_selfish
                else RoundaboutCooperativePolicy(config))
    if st == 'racetrack':
        return (RacetrackSelfishPolicy(config) if is_selfish
                else RacetrackCooperativePolicy(config))

    return SelfishPolicy(config) if is_selfish else CooperativePolicy(config)


