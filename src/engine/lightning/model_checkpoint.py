import os

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointCallback(ModelCheckpoint):
    def __init__(self, *args, save_every_n_epochs=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)
        cur_step = trainer.global_step
        cur_epoch = trainer.current_epoch

        if (
            self.save_every_n_epochs is not None
            and self.save_every_n_epochs > 0
            and (cur_epoch + 1) % self.save_every_n_epochs == 0
        ):
            ckpt_dir = self.dirpath
            assert ckpt_dir is not None
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{cur_epoch+1}.pth")
            state_dict = dict(state_dict=pl_module.state_dict(), epoch=cur_epoch + 1, global_step=cur_step)
            torch.save(state_dict, ckpt_path)

    def save_checkpoint(self, trainer):
        super().save_checkpoint(trainer)

        # * link the best to best.ckpt, not thread safe
        if os.path.isfile(self.best_model_path):
            new_path = os.path.join(os.path.dirname(self.best_model_path), "best.ckpt")
            if os.path.islink(new_path) or os.path.exists(new_path):
                os.unlink(new_path)
            # * use relpath, to make sure link is valid when we move the exp dir
            relpath = os.path.relpath(self.best_model_path, os.path.dirname(new_path))
            os.symlink(relpath, new_path)
