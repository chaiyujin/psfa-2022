from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from datasets.talk_video import samplers
from src.datasets.flame_mesh import FlameMeshDataset

from .utils import build_dataloader


class FlameMeshDataModule(LightningDataModule):
    def __init__(
        self,
        data_src: str,
        dataset_config: DictConfig,
        train_source_voca: List[str],
        valid_source_voca: List[str],
        train_source_coma: List[str],
        valid_source_coma: List[str],
        valid_source_talk: List[str],
        sampler: Optional[str] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        batch_size_coma: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_src = data_src.lower()
        self.config = dataset_config
        self.root = dataset_config.root

        self.sampler = sampler
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        self.batch_size = batch_size
        self.batch_size_coma = batch_size_coma
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_source_voca = train_source_voca
        self.valid_source_voca = valid_source_voca
        self.train_source_coma = train_source_coma
        self.valid_source_coma = valid_source_coma
        self.valid_source_talk = valid_source_talk
        self.trainset_voca: Optional[FlameMeshDataset] = None
        self.validset_voca: Optional[FlameMeshDataset] = None
        self.trainset_coma: Optional[FlameMeshDataset] = None
        self.validset_coma: Optional[FlameMeshDataset] = None
        self.validset_talk: Optional[FlameMeshDataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if len(self.train_source_voca) > 0:
            self.trainset_voca = FlameMeshDataset(self.config, self.root, self.train_source_voca, True)
        if len(self.valid_source_voca) > 0:
            self.validset_voca = FlameMeshDataset(self.config, self.root, self.valid_source_voca, False)
        if len(self.train_source_coma) > 0:
            self.trainset_coma = FlameMeshDataset(self.config, self.root, self.train_source_coma, True)
        if len(self.valid_source_coma) > 0:
            self.validset_coma = FlameMeshDataset(self.config, self.root, self.valid_source_coma, False)
        if len(self.valid_source_talk) > 0:
            self.validset_talk = FlameMeshDataset(self.config, self.root, self.valid_source_talk, False)

    def _build_dataloader(self, dataset, is_trainset, is_coma=False):
        batch_size = self.batch_size_coma if is_coma else self.batch_size
        if (self.sampler is None) or (not is_trainset) or is_coma:
            return build_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=is_trainset,
                collate_fn=dataset.collate,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            sampler_cls = getattr(samplers, self.sampler)
            sampler = sampler_cls(dataset, batch_size=batch_size, shuffle=is_trainset, **self.sampler_kwargs)
            return build_dataloader(
                dataset=dataset,
                batch_sampler=sampler,
                collate_fn=dataset.collate,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

    def train_dataloader(self):
        ret = dict()
        if self.trainset_voca is not None:
            ret["voca"] = self._build_dataloader(self.trainset_voca, True)
        if self.trainset_coma is not None:
            ret["coma"] = self._build_dataloader(self.trainset_coma, True, is_coma=True)
        assert len(ret) > 0
        return ret

    def val_dataloader(self):
        ret = []
        if self.validset_voca is not None:
            ret.append(self._build_dataloader(self.validset_voca, False))
        if self.validset_coma is not None:
            ret.append(self._build_dataloader(self.validset_coma, False, is_coma=True))
        if self.validset_talk is not None:
            ret.append(self._build_dataloader(self.validset_talk, False))
        return ret
