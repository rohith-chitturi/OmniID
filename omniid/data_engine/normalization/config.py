from pydantic import BaseModel
from typing import Dict, Any

class ImageNormalizationConfig(BaseModel):
    width: int = 224
    height: int = 224
    color_space: str = "RGB"
    
class DocumentNormalizationConfig(BaseModel):
    dpi: int = 300
    target_format: str = "JPEG"

class NormalizationConfig(BaseModel):
    """
    Configurable parameters for normalizers to ensure the Data Engine 
    does not assume a single downstream backbone (e.g. DINOv2 vs ViT).
    """
    image: ImageNormalizationConfig = ImageNormalizationConfig()
    document: DocumentNormalizationConfig = DocumentNormalizationConfig()
