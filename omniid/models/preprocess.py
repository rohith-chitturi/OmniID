from pydantic import BaseModel
from typing import List

class PreprocessingSpec(BaseModel):
    resolution: int
    mean: List[float]
    std: List[float]
    interpolation: str = "bicubic"
    color_space: str = "RGB"
