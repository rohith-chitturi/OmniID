class BaseCallback:
    """
    Base class for all TEE Callbacks.
    Priorities dictate execution order (higher priority executes first).
    """
    priority = 0

    def on_execution_start(self, context): pass
    def on_execution_end(self, context): pass
    def on_train_start(self, context): pass
    def on_train_end(self, context): pass
    def on_epoch_start(self, context, epoch): pass
    def on_epoch_end(self, context, epoch): pass
    def on_batch_start(self, context, batch_idx): pass
    def on_batch_end(self, context, batch_idx, loss): pass
    def on_exception(self, context, exception, traceback_str): pass


class CheckpointCallback(BaseCallback):
    priority = 100 # High priority, run first on epoch end
    
    def on_epoch_end(self, context, epoch):
        context.checkpoint_manager.save(
            context=context,
            epoch=epoch,
            global_step=epoch * 100, # Mock step
            model_state={"mock_layer": [0.1, 0.2]}
        )
        # Log structured event
        context.metrics_recorder.log("event_checkpoint_saved", 1, step=epoch)

class ProgressBarCallback(BaseCallback):
    priority = 10 # Lower priority, just printing
    
    def on_epoch_end(self, context, epoch):
        print(f"[Epoch {epoch}] Completed.")
