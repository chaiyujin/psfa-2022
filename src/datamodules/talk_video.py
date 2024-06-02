from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from src.datasets.talk_video import TalkVideoDataset

from .utils import build_dataloader


class TalkVideoDataModule(LightningDataModule):
    def __init__(
        self,
        train_source: List[str],
        valid_source: List[str],
        dataset_config: DictConfig,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.config = dataset_config
        self.root = dataset_config.root

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_source = train_source
        self.valid_source = valid_source
        self.trainset: Optional[TalkVideoDataset] = None
        self.validset: Optional[TalkVideoDataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.trainset = TalkVideoDataset(self.config, root=self.root, csv_source=self.train_source, is_trainset=True)
        self.validset = TalkVideoDataset(self.config, root=self.root, csv_source=self.valid_source, is_trainset=False)

    def train_dataloader(self):
        return build_dataloader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            collate_fn=self.trainset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return build_dataloader(
            dataset=self.validset,
            collate_fn=self.trainset.collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
