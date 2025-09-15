from typing import Union
from policies.selfish_policy import SelfishPolicy
from policies.cooperative_policy import CooperativePolicy
from policies.defensive_policy import DefensivePolicy
from policies.intersection_policy import IntersectionCooperativePolicy, IntersectionSelfishPolicy
from policies.roundabout_policy import RoundaboutCooperativePolicy, RoundaboutSelfishPolicy
from policies.racetrack_policy import RacetrackCooperativePolicy, RacetrackSelfishPolicy
from policies.parking_lot_policy import ParkingLotCooperativePolicy, ParkingLotSelfishPolicy
from policies.official_parking_policy import OfficialParkingCooperativePolicy, OfficialParkingSelfishPolicy
from policies.single_social_law_policy import (
    SingleSocialLawPolicy, SingleSocialLawIntersectionPolicy, 
    SingleSocialLawRoundaboutPolicy, SingleSocialLawRacetrackPolicy
)


def detect_scenario_type(scenario_name: str) -> str:
    name = (scenario_name or '').lower()
    if 'intersection' in name:
        return 'intersection'
    if 'roundabout' in name:
        return 'roundabout'
    if 'racetrack' in name:
        return 'racetrack'
    if 'parking' in name or 'parking_lot' in name:
        return 'parking_lot'
    if 'merge' in name:
        return 'merge'
    if 'highway' in name:
        return 'highway'
    return 'highway'


def create_agent_policy(agent_composition: dict, config: dict, scenario_type: Union[str, None] = None):
    import random as _random
    selfish_ratio = float(agent_composition.get('selfish_ratio', 0.33))
    cooperative_ratio = float(agent_composition.get('cooperative_ratio', 0.33))
    defensive_ratio = float(agent_composition.get('defensive_ratio', 0.34))
    
    # Randomly select ego type according to ratios (seeded upstream)
    rand_val = _random.random()
    if rand_val < selfish_ratio:
        agent_type = 'selfish'
    elif rand_val < selfish_ratio + cooperative_ratio:
        agent_type = 'cooperative'
    else:
        agent_type = 'defensive'
    
    st = (scenario_type or '').strip().lower()

    if st == 'intersection':
        if agent_type == 'selfish':
            return IntersectionSelfishPolicy(config)
        elif agent_type == 'cooperative':
            return IntersectionCooperativePolicy(config)
        else:  # defensive
            return DefensivePolicy(config)
    if st == 'roundabout':
        if agent_type == 'selfish':
            return RoundaboutSelfishPolicy(config)
        elif agent_type == 'cooperative':
            return RoundaboutCooperativePolicy(config)
        else:  # defensive
            return DefensivePolicy(config)
    if st == 'racetrack':
        if agent_type == 'selfish':
            return RacetrackSelfishPolicy(config)
        elif agent_type == 'cooperative':
            return RacetrackCooperativePolicy(config)
        else:  # defensive
            return DefensivePolicy(config)
    if st == 'parking_lot':
        if agent_type == 'selfish':
            return ParkingLotSelfishPolicy(config)
        elif agent_type == 'cooperative':
            return ParkingLotCooperativePolicy(config)
        else:  # defensive
            return DefensivePolicy(config)

    # Default scenarios (highway, merge)
    if agent_type == 'selfish':
        return SelfishPolicy(config)
    elif agent_type == 'cooperative':
        return CooperativePolicy(config)
    else:  # defensive
        return DefensivePolicy(config)


def create_single_social_law_policy(social_law_name: str, config: dict, scenario_type: Union[str, None] = None):
    """
    Create a policy that only applies the specified social law.
    
    Args:
        social_law_name: Name of the social law to apply (e.g., 'cooperative_merging')
        config: Configuration dictionary
        scenario_type: Type of scenario (intersection, roundabout, racetrack, etc.)
        
    Returns:
        Policy instance that applies only the specified social law
        
    Raises:
        ValueError: If the social law name is not found in config
    """
    # Validate social law exists
    if not config or 'social_laws' not in config:
        raise ValueError("No social_laws configuration found")
        
    available_laws = list(config['social_laws'].keys())
    if social_law_name not in available_laws:
        raise ValueError(f"Unknown social law '{social_law_name}'. Available: {available_laws}")
    
    st = (scenario_type or '').strip().lower()
    
    # Create scenario-appropriate policy with single social law
    if st == 'intersection':
        return SingleSocialLawIntersectionPolicy(social_law_name, config)
    elif st == 'roundabout':
        return SingleSocialLawRoundaboutPolicy(social_law_name, config)
    elif st == 'racetrack':
        return SingleSocialLawRacetrackPolicy(social_law_name, config)
    else:
        # Default highway/merge scenarios
        return SingleSocialLawPolicy(social_law_name, config)


def create_official_parking_policy(agent_composition: dict, config: dict):
    """
    Create an official parking policy based on the highway-env parking environment approach.
    
    Args:
        agent_composition: Agent composition dictionary
        config: Configuration dictionary
        
    Returns:
        Official parking policy instance
    """
    import random as _random
    selfish_ratio = float(agent_composition.get('selfish_ratio', 0.33))
    cooperative_ratio = float(agent_composition.get('cooperative_ratio', 0.33))
    defensive_ratio = float(agent_composition.get('defensive_ratio', 0.34))
    
    # Randomly select ego type according to ratios (seeded upstream)
    rand_val = _random.random()
    if rand_val < selfish_ratio:
        agent_type = 'selfish'
    elif rand_val < selfish_ratio + cooperative_ratio:
        agent_type = 'cooperative'
    else:
        agent_type = 'defensive'
    
    if agent_type == 'selfish':
        return OfficialParkingSelfishPolicy(config)
    elif agent_type == 'cooperative':
        return OfficialParkingCooperativePolicy(config)
    else:  # defensive
        return DefensivePolicy(config)


def get_available_social_laws(config: dict) -> list:
    """
    Get list of available social laws from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of social law names
    """
    if not config or 'social_laws' not in config:
        return []
    return list(config['social_laws'].keys())


