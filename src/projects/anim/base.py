import os
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig

from src import constants
from src.data import image as imutils
from src.engine.logging import get_logger
from src.engine.misc import filesys
from src.engine.train import epoch_logging, get_tensorboard_writer, get_trainset, get_validset

from .seq_info import SeqInfo
from .vis import VideoGenerator, get_painters, parse_painter, register_painter

log = get_logger("ANIM")


class PL_AnimBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # * Hparams
        self.save_hyperparameters()
        self.hparams: DictConfig  # type hint
        self.wanna_fps = self.hparams.wanna_fps
        self.seq_info = SeqInfo(1)

        # * Generating videos
        self._generating = False

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                             Logging before start                                             * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        remained = self.trainer.max_epochs - self.current_epoch
        log.info(f"Model has been trained for {self.current_epoch} epoch, {remained} remained.")

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                 Epoch Logging                                                * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_progress_bar_dict(self):
        ret = super().get_progress_bar_dict()
        ret.pop("v_num")
        return ret

    def training_epoch_end(self, outputs):
        epoch_logging(self, outputs)

    def validation_epoch_end(self, outputs):
        epoch_logging(self, outputs)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                  tensorboard                                                 * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def tb_add_images(self, tag, n_plot, batch, results, fi=0, painter=None, debug_mode="each", postfix=""):
        assert debug_mode in ["each", "rows", "cols"]

        hparams = self.hparams
        A: int = hparams.visualizer.draw_grid_size

        # hook before visualization
        if hasattr(self, "on_visualization"):
            self.on_visualization(batch, results)

        if painter is None:
            draw_fns = get_painters(hparams.visualizer)
        else:
            draw_fns = [parse_painter(painter)]

        # select the experiment
        tb = get_tensorboard_writer(self.trainer)
        kwargs = dict(global_step=self.trainer.global_step, dataformats="HWC")
        # add images
        debug_lists = dict()
        for bi in range(n_plot):
            # iter draw functions
            for fn in draw_fns:
                fn_name = fn.__name__ + postfix
                canvas = fn(hparams, batch, results, bi, fi, A=A)
                if canvas is None:
                    continue
                if hparams.debug:
                    if debug_mode == "each":
                        imutils.imwrite(fn_name + ".png", canvas)
                        imutils.imshow(fn_name, canvas)  # type: ignore
                        imutils.waitKey(1)
                    else:
                        if fn_name not in debug_lists:
                            debug_lists[fn_name] = []
                        debug_lists[fn_name].append(canvas)
                # add image
                tb.add_image(f"{tag}-{bi:d}/{fn_name}", canvas, **kwargs)

        if hparams.debug and debug_mode != "each":
            for fn_name, debug_list in debug_lists.items():
                if debug_mode == "rows":
                    debug_im = np.concatenate(debug_list, axis=0)
                elif debug_mode == "cols":
                    debug_im = np.concatenate(debug_list, axis=1)
                else:
                    raise NotImplementedError()
                imutils.imwrite(fn_name + ".png", debug_im)
                imutils.imshow(fn_name, debug_im)
            imutils.waitKey(1)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Generate                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @property
    def generating(self):
        return self._generating

    def to_dump_images(self, results, fi) -> Optional[Dict[str, Any]]:
        return None

    def to_dump_offsets(self, results, fi) -> Optional[Dict[str, Any]]:
        return None

    def generate(
        self,
        epoch,
        global_step,
        generate_dir,
        during_training=False,
        datamodule=None,
        trainset_seq_list=(0,),
        validset_seq_list=(0,),
        extra_tag=None,
    ):
        self._generating = True
        constants.GENERATING = True

        trainset = get_trainset(self, datamodule)
        validset = get_validset(self, datamodule)
        if isinstance(trainset, dict) and trainset.get("talk_video") is not None:
            trainset = trainset["talk_video"]
        elif isinstance(trainset, dict) and trainset.get("voca") is not None:
            trainset = trainset["voca"]
        # assert trainset is None or isinstance(trainset, AnimBaseDataset)
        # assert validset is None or isinstance(validset, AnimBaseDataset)

        media_list = self.parse_media_list()

        draw_fn_list = get_painters(self.hparams.visualizer)
        shared_kwargs = dict(epoch=epoch, media_dir=generate_dir, draw_fn_list=draw_fn_list, extra_tag=extra_tag)

        generator = VideoGenerator(self)

        if self.hparams.visualizer.generate_test:
            generator.generate_for_media(media_list=media_list, **shared_kwargs)

        if self.hparams.visualizer.generate_valid and validset is not None:
            seq_list = self.hparams.visualizer.get("validset_seq_list") or validset_seq_list
            generator.generate_for_dataset(dataset=validset, set_type="valid", seq_list=seq_list, **shared_kwargs)

        if self.hparams.visualizer.generate_train and trainset is not None:
            seq_list = self.hparams.visualizer.get("trainset_seq_list") or trainset_seq_list  # "one-per-speaker"
            generator.generate_for_dataset(dataset=trainset, set_type="train", seq_list=seq_list, **shared_kwargs)

        constants.GENERATING = False
        self._generating = False

    def parse_media_list(self, key_media_list="test_media"):
        test_media = self.hparams.get(key_media_list, dict(do=[]))
        media_list = []

        def str_to_info(media):
            media_path = media
            seq_name, _ = os.path.splitext(os.path.basename(media_path))
            tag = os.path.basename(os.path.dirname(media_path))
            return [tag, seq_name, media_path]

        def append(info):
            if isinstance(info, str):
                info = str_to_info(info)
            if not isinstance(info, list):
                info = list(info)
            if len(info) == 3 and test_media.get("default_extra") is not None:
                info.append(test_media.get("default_extra"))
            media_list.append(info)

        for tag in test_media["do"]:
            src = test_media[tag]
            if isinstance(src, (dict, DictConfig)):
                if "dir" in src:
                    assert "pattern" in src
                    for path in filesys.find_files(src["dir"], src["pattern"], recursive=src.get("recursive", True)):
                        relpath = os.path.relpath(path, src["dir"])
                        append([tag, os.path.splitext(relpath)[0], path])
                elif tag == "vocaset":
                    assert "speakers" in src
                    assert "data_dir" in src
                    data_dir = src["data_dir"]
                    for speaker in src["speakers"]:
                        if isinstance(speaker, str):
                            seq_list = list(range(40))
                        else:
                            seq_list = list(range(*speaker[1]))
                            speaker = speaker[0]
                        spk_dir = os.path.join(data_dir, speaker)
                        for i_seq in seq_list:
                            sent_id = f"sentence{i_seq+1:02d}"
                            append([f"vocaset/{speaker}", sent_id, os.path.join(spk_dir, sent_id)])
            elif isinstance(src, (list, tuple, ListConfig)):
                assert tag == "misc", f"Found {tag} is a list of sources! Should be 'misc'"
                for item in src:
                    if isinstance(item, str):
                        append(item)
                    else:
                        if len(item) >= 5 and item[0] != item[4]:
                            continue
                        append(item)
            elif isinstance(src, str):
                append(src)
        return media_list
