from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TensorBoardLoggerConf:
    _target_: str = "pytorch_lightning.loggers.tensorboard.TensorBoardLogger"

    save_dir: str = "tb/"
    name: Optional[str] = "default"
    version: Any = None  # Optional[Union[int, str]]
    log_graph: bool = False
    default_hp_metric: bool = True
    prefix: str = ""
    sub_dir: Optional[str] = None
