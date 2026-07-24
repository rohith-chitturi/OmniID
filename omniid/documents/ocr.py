from abc import ABC, abstractmethod
from typing import List, Dict, Any

class OCRProvider(ABC):
    """
    Abstract interface for consuming OCR engines like Tesseract, EasyOCR.
    """
    @abstractmethod
    def extract_text(self, image) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts. 
        Each dict must contain:
        - 'text': str
        - 'box': [x1, y1, x2, y2]
        """
        pass

class MockOCRProvider(OCRProvider):
    def extract_text(self, image) -> List[Dict[str, Any]]:
        # Mocking OCR output for testing
        return [
            {"text": "OmniID", "box": [10, 10, 50, 20]},
            {"text": "Document", "box": [60, 10, 120, 20]}
        ]
