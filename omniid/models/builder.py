from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.base import BaseFoundationEncoder

def build_encoder(name: str, **kwargs) -> BaseFoundationEncoder:
    """
    Instantiates an encoder by name from the BACKBONE_REGISTRY.
    """
    # Ensure backbones are imported to trigger registration decorators
    import omniid.models.backbones.dinov2
    
    return BACKBONE_REGISTRY.build(name, **kwargs)
