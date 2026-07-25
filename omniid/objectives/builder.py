from omniid.objectives.registry import OBJECTIVE_REGISTRY
from omniid.objectives.base import BaseObjective

def build_objective(name: str, **kwargs) -> BaseObjective:
    """
    Instantiates an optimization objective strategy by name.
    """
    # Trigger registrations
    import omniid.objectives.strategies.simclr
    import omniid.objectives.strategies.vicreg
    import omniid.objectives.strategies.byol
    import omniid.objectives.strategies.clip
    
    return OBJECTIVE_REGISTRY.build(name, **kwargs)
