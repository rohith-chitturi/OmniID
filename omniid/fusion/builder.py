from omniid.fusion.registry import FUSION_REGISTRY
from omniid.fusion.base import BaseFusionModule

def build_fusion(name: str, **kwargs) -> BaseFusionModule:
    """
    Instantiates a multimodal fusion strategy by name.
    """
    # Trigger registrations
    import omniid.fusion.strategies.concat
    import omniid.fusion.strategies.weighted_sum
    import omniid.fusion.strategies.cross_attention
    import omniid.fusion.strategies.transformer
    
    return FUSION_REGISTRY.build(name, **kwargs)
