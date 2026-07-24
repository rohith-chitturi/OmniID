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
