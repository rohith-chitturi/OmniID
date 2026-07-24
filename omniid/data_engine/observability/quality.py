import cv2
import numpy as np
from typing import List, Dict, Any
from omniid.data_engine.contracts.results import ValidationResult

class QualityAssessor:
    """
    Assesses ML-specific quality (resolution, blur, missing modalities).
    """
    def __init__(self, min_width: int = 100, min_height: int = 100, blur_threshold: float = 100.0):
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold

    def compute_blur(self, image_path: str) -> float:
        # Variance of the Laplacian
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    def assess(self, validated_data: List[Dict[str, Any]]) -> ValidationResult:
        result = ValidationResult()
        
        for sample in validated_data:
            identity_id = sample["id"]
            valid = True
            
            modalities = sample.get("modalities", {})
            if "face" not in modalities:
                result.add_warning(identity_id, "Missing face modality.")
            
            for mod, path_str in modalities.items():
                if path_str.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(path_str)
                    if img is None:
                        continue # Should have been caught by validator
                    
                    h, w = img.shape[:2]
                    if w < self.min_width or h < self.min_height:
                        result.add_error(identity_id, f"Resolution too low: {w}x{h} < {self.min_width}x{self.min_height}")
                        valid = False
                        
                    blur = self.compute_blur(path_str)
                    if blur < self.blur_threshold:
                        result.add_warning(identity_id, f"Image may be blurred (Laplacian var: {blur:.2f})")
                        
            if valid:
                result.accepted.append(sample)
            else:
                result.rejected.append(sample)

        return result
