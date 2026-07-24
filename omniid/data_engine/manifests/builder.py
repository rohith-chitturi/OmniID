import json
import os
from datetime import datetime
from typing import List, Dict, Any
from omniid.data_engine.contracts.results import ValidationResult

class ManifestBuilder:
    """
    Builds the formal dataset manifest (JSON) and the human-readable report.
    """
    def __init__(self, dataset_name: str, version: str = "1.0.0"):
        self.dataset_name = dataset_name
        self.version = version
        self.schema_version = "1.0.0"
        self.pipeline_version = "1.0.0"

    def generate(self, normalized_data: List[Dict[str, Any]], fingerprint: str, validation_result: ValidationResult, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Gather Modalities and Stats
        modalities_set = set()
        missing_voice_count = 0
        face_count = 0
        doc_count = 0
        sig_count = 0
        
        for sample in normalized_data:
            mods = sample.get("modalities", {})
            for m in mods.keys():
                modalities_set.add(m)
                
            if "face" in mods: face_count += 1
            if "document" in mods: doc_count += 1
            if "signature" in mods: sig_count += 1
            if "voice" not in mods: missing_voice_count += 1

        statistics = {
            "total_samples": len(normalized_data),
            "modalities_present": list(modalities_set),
            "face_count": face_count,
            "document_count": doc_count,
            "signature_count": sig_count,
            "missing_voice": missing_voice_count,
            "validation_rejected": len(validation_result.rejected),
            "validation_errors": len(validation_result.errors),
            "validation_warnings": len(validation_result.warnings)
        }
        
        # 2. JSON Manifest
        manifest = {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "fingerprint": fingerprint,
            "statistics": statistics
        }
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # 3. Human-Readable Report
        report_lines = [
            f"Dataset Report: {self.dataset_name}",
            f"Version: {self.version}",
            f"Created At: {manifest['created_at']}",
            f"Fingerprint: {fingerprint}",
            "-" * 40,
            f"Samples: {statistics['total_samples']}",
            f"Faces: {statistics['face_count']}",
            f"Documents: {statistics['document_count']}",
            f"Signatures: {statistics['signature_count']}",
            f"Missing Voice: {statistics['missing_voice']}",
            f"Rejected: {statistics['validation_rejected']}",
            f"Warnings/Blur/Issues: {statistics['validation_warnings']}",
            "-" * 40,
            "End of Report"
        ]
        
        report_path = os.path.join(output_dir, "dataset_report.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

        return manifest_path
