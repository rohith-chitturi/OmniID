from pydantic import BaseModel
from typing import List, Optional

class PreprocessingSpec(BaseModel):
    resolution: int
    mean: List[float]
    std: List[float]
    interpolation: str = "bicubic"
    color_space: str = "RGB"

class DocumentPreprocessingSpec(PreprocessingSpec):
    bounding_box_scale: Optional[int] = None
    requires_ocr: bool = False
    max_pages: int = 1
