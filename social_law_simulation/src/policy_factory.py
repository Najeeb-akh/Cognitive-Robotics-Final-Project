from policies.selfish_policy import SelfishPolicy
from policies.cooperative_policy import CooperativePolicy
from policies.intersection_policy import IntersectionCooperativePolicy, IntersectionSelfishPolicy
from policies.roundabout_policy import RoundaboutCooperativePolicy, RoundaboutSelfishPolicy
from policies.racetrack_policy import RacetrackCooperativePolicy, RacetrackSelfishPolicy
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


def create_single_social_law_policy(social_law_name: str, config: dict, scenario_type: str | None = None):
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


def get_available_social_laws(config: dict) -> list[str]:
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


