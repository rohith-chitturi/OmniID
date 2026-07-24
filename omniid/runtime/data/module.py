from abc import ABC, abstractmethod

class BaseDataModule(ABC):
    """
    Lightweight abstraction over datasets and dataloaders.
    Mirrors proven Lightning-style interfaces.
    """
    @abstractmethod
    def prepare_data(self):
        """Run once (e.g., download, fingerprint)."""
        pass

    @abstractmethod
    def setup(self, stage: str):
        """Setup train/val/test splits."""
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

class IdentityDataModule(BaseDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return [1, 2, 3]

    def val_dataloader(self):
        return [1, 2]
