import os
import json
from pathlib import Path
from typing import Dict, Any, List

class IngestionParser:
    """
    Parses a raw dataset directory into a structured dictionary of identity samples.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def parse(self) -> Dict[str, Any]:
        """
        Parses the dataset directory.
        Returns a dictionary mapping identity_id to their collected modalities and metadata.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")

        metadata_path = self.dataset_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    pass # We will catch this in validation

        identities: Dict[str, Dict[str, Any]] = {}
        
        # Scrape modalities
        for modality in ["face", "document", "signature", "voice"]:
            modality_dir = self.dataset_path / modality
            if modality_dir.exists() and modality_dir.is_dir():
                for file_path in modality_dir.iterdir():
                    if file_path.is_file():
                        # Assume filename is the identity_id, e.g., "id123.jpg" -> "id123"
                        identity_id = file_path.stem
                        if identity_id not in identities:
                            identities[identity_id] = {"id": identity_id, "modalities": {}}
                        identities[identity_id]["modalities"][modality] = str(file_path)

        # Attach metadata
        if isinstance(metadata, dict):
            for identity_id, meta in metadata.items():
                if identity_id not in identities:
                    identities[identity_id] = {"id": identity_id, "modalities": {}}
                identities[identity_id]["metadata"] = meta

        return identities
