import pytest
from omegaconf import OmegaConf
from omniid.experiment.config import fingerprint_config, diff_configs
from omniid.experiment.registry import ExperimentRegistry
import os
import shutil

def test_fingerprint_determinism():
    cfg1 = OmegaConf.create({"a": 1, "b": {"c": 2}})
    cfg2 = OmegaConf.create({"b": {"c": 2}, "a": 1})
    assert fingerprint_config(cfg1) == fingerprint_config(cfg2)

def test_diff_configs():
    cfg1 = OmegaConf.create({"model": {"lr": 0.01}, "batch": 32})
    cfg2 = OmegaConf.create({"model": {"lr": 0.05}, "batch": 32})
    diff = diff_configs(cfg1, cfg2)
    
    assert "model.lr" in diff
    assert diff["model.lr"]["old"] == 0.01
    assert diff["model.lr"]["new"] == 0.05
    assert "batch" not in diff

def test_experiment_registry():
    test_dir = "tests/test_experiments"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    registry = ExperimentRegistry(base_dir=test_dir)
    cfg = OmegaConf.create({"seed": 42, "tags": ["test"]})
    
    # Create first run
    exp_id, run_id, run_dir = registry.create_run(cfg, experiment_name="baseline")
    assert exp_id == "baseline"
    assert run_id == "RUN-0001"
    assert os.path.exists(os.path.join(run_dir, "config.yaml"))
    assert os.path.exists(os.path.join(run_dir, "config_fingerprint.txt"))
    assert os.path.exists(os.path.join(run_dir, "environment.json"))
    assert os.path.exists(os.path.join(run_dir, "artifact_manifest.json"))
    assert os.path.exists(os.path.join(run_dir, "run_metadata.json"))
    
    # Create second run in same experiment
    exp_id2, run_id2, run_dir2 = registry.create_run(cfg, experiment_name="baseline")
    assert exp_id2 == "baseline"
    assert run_id2 == "RUN-0002"
