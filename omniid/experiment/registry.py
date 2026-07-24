import os
import json
import uuid
import datetime
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple
from omniid.experiment.models import RunMetadata, ExperimentState, RunArtifactManifest
from omniid.experiment.snapshot import capture_environment
from omniid.experiment.config import fingerprint_config

class ExperimentRegistry:
    """
    Manages the lifecycle of Experiments and Runs.
    Resolves unique identifiers (EXP-0001, RUN-0001) and saves reproducible metadata.
    """
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_next_id(self, prefix: str, directory: str) -> str:
        if not os.path.exists(directory):
            return f"{prefix}-0001"
        existing = [d for d in os.listdir(directory) if d.startswith(f"{prefix}-")]
        if not existing:
            return f"{prefix}-0001"
        nums = [int(d.split('-')[1]) for d in existing]
        return f"{prefix}-{max(nums) + 1:04d}"

    def create_run(self, cfg: DictConfig, experiment_name: str = "auto") -> Tuple[str, str, str]:
        # Resolve Experiment ID
        if experiment_name == "auto" or experiment_name is None:
            exp_id = self._get_next_id("EXP", self.base_dir)
        else:
            exp_id = experiment_name
            
        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Resolve Run ID
        run_id = self._get_next_id("RUN", exp_dir)
        run_dir = os.path.join(exp_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        
        # Capture Snapshot & Config Fingerprint
        env_snapshot = capture_environment()
        cfg_fingerprint = fingerprint_config(cfg)
        
        # Write Config YAML
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
            
        # Write Config Fingerprint
        with open(os.path.join(run_dir, "config_fingerprint.txt"), "w") as f:
            f.write(cfg_fingerprint)
            
        # Write Environment
        with open(os.path.join(run_dir, "environment.json"), "w") as f:
            f.write(env_snapshot.model_dump_json(indent=2))

        # Determine Tags
        tags = list(cfg.get("tags", []))
        seed = cfg.get("seed", 42)

        # Write Artifact Manifest
        artifacts = RunArtifactManifest()
        with open(os.path.join(run_dir, "artifact_manifest.json"), "w") as f:
            f.write(artifacts.model_dump_json(indent=2))

        # Write Run Metadata
        run_meta = RunMetadata(
            run_id=run_id,
            experiment_id=exp_id,
            state=ExperimentState.CREATED,
            tags=tags,
            seed=seed,
            artifacts=artifacts
        )
        
        with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
            f.write(run_meta.model_dump_json(indent=2))

        return exp_id, run_id, run_dir
