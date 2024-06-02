from copy import deepcopy
from typing import Optional

from omegaconf import DictConfig, open_dict
from pytorch_lightning import LightningDataModule

from src.datasets.base.samplers import LitmitedFramesBatchSampler
from src.datasets.flame_mesh import FlameMeshDataset
from src.datasets.talk_video import TalkVideoDataset

from .utils import build_dataloader


class TalkAndFlameDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_config: DictConfig,
        train_source_talk_video,
        valid_source_talk_video,
        train_source_flame_mesh,
        valid_source_flame_mesh,
        batch_size,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # configs
        self.cfg_talk_video = deepcopy(dataset_config.talk_video)
        self.cfg_flame_mesh = deepcopy(dataset_config.flame_mesh)
        with open_dict(self.cfg_talk_video):
            self.cfg_talk_video.style_ids = dataset_config.style_ids
        with open_dict(self.cfg_flame_mesh):
            self.cfg_flame_mesh.style_ids = dataset_config.style_ids

        self.trainset_talk_video: Optional[TalkVideoDataset] = None
        self.validset_talk_video: Optional[TalkVideoDataset] = None
        self.trainset_flame_mesh: Optional[FlameMeshDataset] = None

        self.train_source_talk_video = train_source_talk_video
        self.valid_source_talk_video = valid_source_talk_video
        self.train_source_flame_mesh = train_source_flame_mesh
        self.valid_source_flame_mesh = valid_source_flame_mesh

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.trainset_talk_video = TalkVideoDataset(
            config=self.cfg_talk_video,
            root=self.cfg_talk_video.root,
            csv_source=self.train_source_talk_video,
            is_trainset=True,
        )
        self.validset_talk_video = TalkVideoDataset(
            config=self.cfg_talk_video,
            root=self.cfg_talk_video.root,
            csv_source=self.valid_source_talk_video,
            is_trainset=False,
        )
        self.trainset_flame_mesh = FlameMeshDataset(
            config=self.cfg_flame_mesh,
            root=self.cfg_flame_mesh.root,
            csv_source=self.train_source_flame_mesh,
            is_trainset=True,
        )

    def train_dataloader(self):
        max_frames = len(self.trainset_talk_video)
        return dict(
            talk_video=build_dataloader(
                dataset=self.trainset_talk_video,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            ),
            flame_mesh=build_dataloader(
                dataset=self.trainset_flame_mesh,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                batch_sampler=LitmitedFramesBatchSampler(
                    self.trainset_flame_mesh, max_frames, self.batch_size, shuffle=True
                ),
            ),
        )

    def val_dataloader(self):
        return build_dataloader(
            dataset=self.validset_talk_video,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
