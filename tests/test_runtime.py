import pytest
import os
import shutil
from omniid.runtime.core.context import RuntimeContext, TrainingState
from omniid.runtime.core.engine import ExecutionEngine
from omniid.runtime.hardware.managers import DeviceManager, PrecisionManager
from omniid.runtime.checkpointing.manager import CheckpointManager
from omniid.runtime.callbacks.base import CheckpointCallback, ProgressBarCallback
from omniid.runtime.trainer.base import PlaceholderTrainer, PlaceholderEncoder
from omniid.runtime.data.module import IdentityDataModule
from omniid.experiment.metrics import JSONMetricsRecorder

def test_execution_engine_lifecycle():
    run_dir = "tests/test_run_dir"
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    
    # Mock services
    metrics = JSONMetricsRecorder(run_dir)
    checkpointer = CheckpointManager(run_dir)
    device = DeviceManager()
    precision = PrecisionManager()
    
    context = RuntimeContext(
        experiment_id="EXP-TEST",
        run_id="RUN-TEST",
        config_fingerprint="abc123hash",
        metrics_recorder=metrics,
        checkpoint_manager=checkpointer,
        device_manager=device,
        precision_manager=precision,
        seed=42
    )
    
    engine = ExecutionEngine(
        context=context,
        trainer_cls=PlaceholderTrainer,
        model=PlaceholderEncoder(),
        datamodule=IdentityDataModule()
    )
    
    engine.register_callback(CheckpointCallback())
    engine.register_callback(ProgressBarCallback())
    
    assert engine.state == TrainingState.INITIALIZED
    
    engine.execute()
    
    assert engine.state == TrainingState.COMPLETED
    
    # Verify outputs
    metrics.save()
    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "checkpoints", "checkpoint_epoch_003.json"))
