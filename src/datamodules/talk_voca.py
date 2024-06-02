from typing import Any, Dict, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from ..datasets.talk_voca import samplers
from ..datasets.talk_voca.talk_voca import TalkVOCADataset
from .utils import build_dataloader


class TalkVOCADataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DictConfig,
        train_source_talk,
        valid_source_talk,
        train_source_voca,
        valid_source_voca,
        batch_size,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler: Optional[str] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sampler = sampler
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs

        # configs
        self.cfg = dataset_config

        self.trainset: Optional[TalkVOCADataset] = None
        self.validset: Optional[TalkVOCADataset] = None

        self.train_source_talk = train_source_talk
        self.valid_source_talk = valid_source_talk
        self.train_source_voca = train_source_voca
        self.valid_source_voca = valid_source_voca

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.trainset = TalkVOCADataset(
            config=self.cfg,
            root_talk=self.cfg.root_talk,
            root_voca=self.cfg.root_voca,
            csv_talk=self.train_source_talk,
            csv_voca=self.train_source_voca,
            is_trainset=True,
        )
        self.validset = TalkVOCADataset(
            config=self.cfg,
            root_talk=self.cfg.root_talk,
            root_voca=self.cfg.root_voca,
            csv_talk=self.valid_source_talk,
            csv_voca=self.valid_source_voca,
            is_trainset=False,
        )

    def _build_dataloader(self, dataset, is_trainset):
        if (self.sampler is None) or (not is_trainset):
            return build_dataloader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=is_trainset,
                collate_fn=dataset.collate,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=is_trainset,
            )
        else:
            sampler_cls = getattr(samplers, self.sampler)
            sampler = sampler_cls(dataset, batch_size=self.batch_size, shuffle=is_trainset, **self.sampler_kwargs)
            return build_dataloader(
                dataset=dataset,
                batch_sampler=sampler,
                collate_fn=dataset.collate,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def train_dataloader(self):
        return self._build_dataloader(self.trainset, True)

    def val_dataloader(self):
        if len(self.validset) == 0:
            return None

        dataloader = build_dataloader(
            dataset=self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.validset.collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
