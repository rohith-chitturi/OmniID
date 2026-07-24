import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any
from omniid.data_engine.contracts.results import ValidationResult

class DatasetValidator:
    """
    Validates ingested data against contracts (existence, corruption, schema).
    Does NOT assess ML quality (e.g., blur).
    """
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.wav'}

    def validate(self, ingested_data: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()

        for identity_id, sample in ingested_data.items():
            valid = True
            
            # Check modalities
            modalities = sample.get("modalities", {})
            if not modalities:
                result.add_error(identity_id, "No modalities found for identity.")
                valid = False
            
            for mod, path_str in modalities.items():
                p = Path(path_str)
                # 1. Existence
                if not p.exists():
                    result.add_error(identity_id, f"File {path_str} does not exist.")
                    valid = False
                    continue
                
                # 2. Supported Format
                if p.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    result.add_error(identity_id, f"Unsupported format: {p.suffix}")
                    valid = False
                    continue
                
                # 3. Corruption Check (for images)
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                    except (UnidentifiedImageError, IOError):
                        result.add_error(identity_id, f"Corrupted image file: {path_str}")
                        valid = False

            # Check Metadata Schema
            meta = sample.get("metadata")
            if not meta:
                result.add_warning(identity_id, "Missing metadata.")
            elif not isinstance(meta, dict):
                result.add_error(identity_id, "Metadata must be a dictionary.")
                valid = False
            else:
                if "age" in meta and not isinstance(meta["age"], int):
                    result.add_error(identity_id, "Metadata 'age' must be integer.")
                    valid = False

            if valid:
                result.accepted.append(sample)
            else:
                result.rejected.append(sample)

        return result
