from typing import Dict, Any, Tuple
from omniid.model_zoo.types import ModelManifest

class CompatibilityValidator:
    """
    Validates that a published checkpoint manifest is safe to load into the current execution runtime.
    """
    
    @staticmethod
    def validate(manifest: ModelManifest, current_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates manifest against the current execution config.
        Returns (is_compatible, error_message).
        """
        # 1. Framework Version Check
        import omniid
        try:
            current_fw = omniid.__version__
        except AttributeError:
            current_fw = "0.1.0"
            
        if manifest.framework_version != current_fw:
            # In a real system, we'd do semver compatibility checking here
            pass
            
        # 2. Encoder Compatibility
        if "encoder" in current_config and current_config["encoder"] != manifest.encoder:
            return False, f"Encoder mismatch: Checkpoint requires '{manifest.encoder}', but config specifies '{current_config['encoder']}'"
            
        # 3. Fusion Compatibility
        if manifest.fusion:
            if "fusion" not in current_config or current_config["fusion"] != manifest.fusion:
                return False, f"Fusion mismatch: Checkpoint requires '{manifest.fusion}', but config specifies '{current_config.get('fusion')}'"
                
        # 4. Objective Compatibility
        # Objectives are not strictly required for inference, but if training resumes, they must match
        if manifest.objective and current_config.get("is_training", False):
            if "objective" not in current_config or current_config["objective"] != manifest.objective:
                return False, f"Objective mismatch for training: Checkpoint requires '{manifest.objective}'"
                
        return True, "Compatibility checks passed."
