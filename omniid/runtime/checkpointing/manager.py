import os
import json

class CheckpointManager:
    """
    Serializes full run state including experiment metadata and config fingerprints.
    """
    def __init__(self, run_dir: str):
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.framework_version = "0.1.0"

    def save(self, context, epoch: int, global_step: int, model_state: dict):
        ckpt_name = f"checkpoint_epoch_{epoch:03d}.json" # JSON for mock purposes right now
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        
        bundle = {
            "metadata": {
                "checkpoint_version": 1,
                "framework_version": self.framework_version,
                "experiment_id": context.experiment_id,
                "run_id": context.run_id,
                "config_fingerprint": context.config_fingerprint,
                "dataset_fingerprint": "mock_dataset_fp_123", # Normally extracted from datamodule
                "global_step": global_step,
                "epoch": epoch
            },
            "model_state_dict": model_state,
            "optimizer_state_dict": {},
            "scheduler_state_dict": {}
        }
        
        with open(ckpt_path, "w") as f:
            json.dump(bundle, f, indent=2)
            
        return ckpt_path
