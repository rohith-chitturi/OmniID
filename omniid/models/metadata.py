from pydantic import BaseModel
from typing import Optional

class ModelMetadata(BaseModel):
    name: str
    architecture: str
    input_resolution: int
    patch_size: int
    embedding_dim: int
    parameter_count: int
    pretrained_source: str
    license: Optional[str] = None
    
    # Document Capabilities
    supports_layout: bool = False
    supports_ocr: bool = False
    supports_generation: bool = False
    max_sequence_length: Optional[int] = None
    supported_languages: Optional[list[str]] = None
