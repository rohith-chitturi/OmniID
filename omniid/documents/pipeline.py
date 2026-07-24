from typing import Dict, Any, List
from omniid.documents.ocr import OCRProvider
from omniid.models.preprocess import DocumentPreprocessingSpec

class DocumentPreprocessingPipeline:
    """
    Isolates Layout normalization, Page rendering hooks, and OCR consumption
    away from the Neural Network encoders.
    """
    def __init__(self, spec: DocumentPreprocessingSpec, ocr_provider: OCRProvider = None):
        self.spec = spec
        self.ocr_provider = ocr_provider

    def normalize_bounding_boxes(self, original_boxes: List[List[int]], width: int, height: int) -> List[List[int]]:
        """
        Scales arbitrary [x1, y1, x2, y2] to the target scale (e.g. 0-1000).
        """
        if self.spec.bounding_box_scale is None:
            return original_boxes
            
        scale = self.spec.bounding_box_scale
        normalized = []
        for box in original_boxes:
            nx1 = int((box[0] / width) * scale)
            ny1 = int((box[1] / height) * scale)
            nx2 = int((box[2] / width) * scale)
            ny2 = int((box[3] / height) * scale)
            # Clamp to [0, scale]
            normalized.append([
                max(0, min(nx1, scale)),
                max(0, min(ny1, scale)),
                max(0, min(nx2, scale)),
                max(0, min(ny2, scale))
            ])
        return normalized

    def __call__(self, image: Any) -> Dict[str, Any]:
        """
        Executes the preprocessing logic to output tensors ready for encode().
        """
        result = {"image_tensor": "mock_tensor"} # Mock tensor logic
        
        if self.spec.requires_ocr:
            if not self.ocr_provider:
                raise ValueError("OCR required by spec but no OCRProvider was passed.")
                
            ocr_results = self.ocr_provider.extract_text(image)
            
            raw_boxes = [r["box"] for r in ocr_results]
            text = [r["text"] for r in ocr_results]
            
            # Assuming mock width=100, height=100 for the original image for this demo
            norm_boxes = self.normalize_bounding_boxes(raw_boxes, width=100, height=100)
            
            result["layout_boxes"] = norm_boxes
            result["text_tokens"] = text
            
        return result
