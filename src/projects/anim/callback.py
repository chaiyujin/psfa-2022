from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule, Trainer

from src.engine.logging import get_logger
from src.engine.misc.table import Table

log = get_logger(__name__)


def log_data_info_once(pl_module):
    data_info = pl_module.hparams.get("data_info")
    if data_info is None:
        return
    table = Table(list_mode="compact", dict_mode="multi-line")
    for key, val in data_info.items():
        if isinstance(val, (dict, DictConfig)):
            table.divide()
        table.add_row(str(key), val)
        if isinstance(val, (dict, DictConfig)):
            table.divide()
    log.info("Data Info:\n{}".format(table))


def log_loss_information_once(pl_module):
    loss_opts = pl_module.hparams.get("loss")
    if loss_opts is None:
        return
    table = Table(list_mode="compact", dict_mode="multi-line")
    for key, val in loss_opts.items():
        if isinstance(val, (dict, DictConfig)):
            table.divide()
        table.add_row(str(key), val)
        if isinstance(val, (dict, DictConfig)):
            table.divide()
    log.info("Loss Options:\n{}".format(table))


class PL_AnimCallback(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        log.info("Model FPS:\n>> {} fps".format(pl_module.wanna_fps))
        log_data_info_once(pl_module)
        log_loss_information_once(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            return

        hparams = pl_module.hparams
        if hparams.get("visualizer") is None:
            return

        cur_epoch = trainer.current_epoch
        gap_epoch = hparams.visualizer.video_gap_epochs
        max_epochs = trainer.max_epochs

        if (
            (hparams.visualizer.video_1st_epoch and cur_epoch == 0)
            or (gap_epoch > 0 and (cur_epoch + 1) % gap_epoch == 0)  # 1st epoch
            or (cur_epoch + 1 == max_epochs)  # gap epoch  # last epoch
        ):
            pl_module.generate(
                epoch=cur_epoch + 1,
                global_step=trainer.global_step,
                generate_dir="media",
                during_training=True,
            )
