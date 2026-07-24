import os
from PIL import Image
from typing import List, Dict, Any
from omniid.data_engine.normalization.config import NormalizationConfig

class DatasetNormalizer:
    """
    Normalizes datasets into a consistent baseline (RGB, dimension resizing) based on config.
    Saves outputs to a normalized artifacts directory.
    """
    def __init__(self, config: NormalizationConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def normalize(self, quality_assessed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_data = []

        for sample in quality_assessed_data:
            identity_id = sample["id"]
            new_sample = {"id": identity_id, "modalities": {}, "metadata": sample.get("metadata", {})}
            
            for mod, path_str in sample.get("modalities", {}).items():
                if path_str.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        with Image.open(path_str) as img:
                            if self.config.image.color_space == "RGB":
                                img = img.convert("RGB")
                            
                            # Configurable resizing
                            img = img.resize((self.config.image.width, self.config.image.height))
                            
                            out_path = os.path.join(self.output_dir, f"{identity_id}_{mod}.jpg")
                            img.save(out_path, "JPEG")
                            new_sample["modalities"][mod] = out_path
                    except Exception as e:
                        pass # Skipping for now if something unexpected happens during write

            normalized_data.append(new_sample)

        return normalized_data
