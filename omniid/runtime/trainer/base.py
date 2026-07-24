from abc import ABC, abstractmethod
from omniid.runtime.core.events import EventBus
from omniid.runtime.core.context import RuntimeContext

class BaseTrainer(ABC):
    """
    Hook-based ML Lifecycle orchestrator.
    Emits events to the EventBus. Defines fit/validate/test structure.
    """
    def __init__(self, context: RuntimeContext, event_bus: EventBus):
        self.context = context
        self.event_bus = event_bus

    def fit(self, model, datamodule):
        self.event_bus.dispatch("on_train_start", self.context)
        
        # Apply hardware abstractions
        model = self.context.precision_manager.apply(model)
        
        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        
        for epoch in range(1, 4): # Hardcoded epochs for now
            self.training_epoch(model, train_loader, epoch)
            
        self.event_bus.dispatch("on_train_end", self.context)

    def training_epoch(self, model, dataloader, epoch: int):
        self.event_bus.dispatch("on_epoch_start", self.context, epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            self.training_step(model, batch, batch_idx)
            
        self.event_bus.dispatch("on_epoch_end", self.context, epoch)

    def training_step(self, model, batch, batch_idx: int):
        self.event_bus.dispatch("on_batch_start", self.context, batch_idx)
        loss = self.optimizer_step(model, batch)
        self.event_bus.dispatch("on_batch_end", self.context, batch_idx, loss)

    @abstractmethod
    def optimizer_step(self, model, batch):
        pass


class PlaceholderTrainer(BaseTrainer):
    """
    Mock trainer used to test the TEE infrastructure before backbones arrive.
    """
    def optimizer_step(self, model, batch):
        # Mock forward & backward
        _ = model(batch)
        loss = 0.5
        self.context.metrics_recorder.log("train_loss", loss)
        return loss

class PlaceholderEncoder:
    """Mock neural network."""
    def __call__(self, x):
        return x
