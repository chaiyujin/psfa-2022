from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"

    checkpoint_callback: Optional[bool] = None  # depercated in 1.5
    enable_checkpointing: bool = True
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    gpus: Any = None
    progress_bar_refresh_rate: Optional[int] = None
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool] = False
    accumulate_grad_batches: Any = None  # Optional[Union[int, Dict[int, int]]] = None,
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Any = None  # Optional[Union[str, timedelta, Dict[str, int]]] = None
    flush_logs_every_n_steps: Optional[int] = None
    log_every_n_steps: int = 50
    num_sanity_val_steps: int = 2
    benchmark: bool = False
    deterministic: bool = False
    weights_summary: Optional[str] = "top"
