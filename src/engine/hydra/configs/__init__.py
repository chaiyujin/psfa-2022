from hydra.core.config_store import ConfigStore

from src.libmorph.config import FLAMEConfig

from .model_checkpoint import ModelCheckpointConf
from .path import PathConf
from .tensorboard import TensorBoardLoggerConf
from .trainer import TrainerConf


def store_configs():
    cs = ConfigStore.instance()
    cs.store(group="path", name="base_path", node=PathConf)
    cs.store(group="trainer", name="base_trainer", node=TrainerConf)
    cs.store(group="callbacks/model_checkpoint", name="base_model_checkpoint", node=ModelCheckpointConf)
    cs.store(group="logger/tensorboard", name="base_tensorboard_logger", node=TensorBoardLoggerConf)
    cs.store(group="flame", name="base_flame", node=FLAMEConfig)
