import hashlib
import json
from typing import List, Dict, Any

class DatasetFingerprinter:
    """
    Generates a deterministic cryptographic fingerprint for a dataset version.
    Includes data payload, schema version, and configuration state.
    """
    def __init__(self, schema_version: str = "1.0.0", pipeline_version: str = "1.0.0"):
        self.schema_version = schema_version
        self.pipeline_version = pipeline_version

    def compute(self, normalized_data: List[Dict[str, Any]], config_dict: Dict[str, Any]) -> str:
        # Sort to ensure determinism
        sorted_data = sorted(normalized_data, key=lambda x: x["id"])
        
        fingerprint_payload = {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "config": config_dict,
            "data": sorted_data
        }
        
        payload_str = json.dumps(fingerprint_payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
