import traceback
from omniid.runtime.core.context import RuntimeContext, TrainingState
from omniid.runtime.core.events import EventBus

class ExecutionEngine:
    """
    Wraps the BaseTrainer, managing the global execution lifecycle, 
    failure recovery, and exception emission.
    """
    def __init__(self, context: RuntimeContext, trainer_cls, model, datamodule):
        self.context = context
        self.event_bus = EventBus()
        self.trainer = trainer_cls(self.context, self.event_bus)
        self.model = model
        self.datamodule = datamodule
        self.state = TrainingState.INITIALIZED

    def register_callback(self, callback):
        self.event_bus.register(callback)

    def execute(self):
        try:
            self.state = TrainingState.TRAINING
            self.event_bus.dispatch("on_execution_start", self.context)
            
            # Execute Training
            self.trainer.fit(self.model, self.datamodule)
            
            self.state = TrainingState.COMPLETED
            self.event_bus.dispatch("on_execution_end", self.context)
        except Exception as e:
            self.state = TrainingState.FAILED
            # Save failure metadata and emit exception event
            self.event_bus.dispatch("on_exception", self.context, e, traceback.format_exc())
            print(f"ExecutionEngine intercepted failure: {e}")
            raise
