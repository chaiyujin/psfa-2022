from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelCheckpointConf:
    _target_: str = "src.engine.lightning.model_checkpoint.ModelCheckpointCallback"

    dirpath: Optional[str] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Any = None  # Optional[timedelta]
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None

    # custom
    save_every_n_epochs: Optional[int] = None
