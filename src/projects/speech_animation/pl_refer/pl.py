from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, open_dict

from assets import FACE_LOWER_VIDX, FACE_NEYE_VIDX, FACE_NOEYEBALLS_VIDX, LIPS_VIDX
from src import metrics
from src.data.image import may_CHW2HWC
from src.engine import ops
from src.engine.builder import load_or_build
from src.engine.logging import get_logger
from src.engine.misc import dotdict, run_once
from src.engine.seq_ops import unfold
from src.engine.train import get_optim_and_lrsch, print_dict, reduce_dict, sum_up_losses
from src.mm_fitting.libmorph import LossModule
from src.modules.neural_renderer import NeuralRenderer
from src.projects.anim import PL_AnimBase, SeqInfo

from ..loss import compute_mesh_loss
from . import visualize as _
from .model import AnimNetRefer

log = get_logger("AnimNetRefer")
FACE_VIDX = FACE_NOEYEBALLS_VIDX
NON_FACE_VIDX = [x for x in range(5023) if x not in FACE_VIDX]


class PL_AnimNetRefer(PL_AnimBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # * AnimNet: Audio -> 3D Animation (Coefficients or Vertex-Offsets)
        self.animnet = load_or_build(self.hparams.animnet, AnimNetRefer)
        # maybe frozen
        self.freeze_animnet = self.hparams.freeze_animnet
        if self.freeze_animnet:
            ops.freeze(self.animnet)

        # * Seq info
        self.seq_info = SeqInfo(self.hparams.animnet.src_seq_frames, self.hparams.animnet.src_seq_pads)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.animnet.state_dict(destination, prefix + "animnet.", keep_vars)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                          Specific Methods for Module                                         * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def to_dump_offsets(self, results, fi) -> Optional[Dict[str, Any]]:
        return {"final": results["animnet"]["dfrm_delta"][0, fi]}

    @torch.no_grad()
    def forward(self, batch, **kwargs):
        self.eval()
        return self.run_animnet(batch, **kwargs)

    # The batch may contain mulitple dataset
    def get_data_for_animnet(self, batch):
        return batch

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                    AnimNet                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def run_animnet(self, batch, **kwargs):
        adata = self.get_data_for_animnet(batch)

        # * inputs
        idle_verts = adata.get("idle_verts")
        audio_dict = adata.get("audio_dict", dict())
        assert idle_verts is not None

        # * animnet model
        stylized_out = self.animnet(
            idle_verts=idle_verts,
            audio_dict=audio_dict,
            refer_offsets=batch.get("offsets_swap_style"),
            **kwargs,
        )
        if self.generating:
            if "offsets_swap_style" in batch:
                batch.pop("offsets_swap_style")
        code_dict = stylized_out["code_dict"]

        # prepare the output verts_dict
        idle_verts = idle_verts.unsqueeze(1)
        trck_delta = batch.get("offsets_tracked")
        REAL_delta = batch.get("offsets_REAL")

        out = dict(
            idle_verts=idle_verts,
            trck_verts=(idle_verts + trck_delta) if trck_delta is not None else None,
            REAL_verts=(idle_verts + REAL_delta) if REAL_delta is not None else None,
            dfrm_verts=stylized_out["out_verts"],
            trck_delta=trck_delta,
            REAL_delta=REAL_delta,
            dfrm_delta=stylized_out["out_verts"] - idle_verts,
            code_dict=code_dict,
        )

        # print_dict("out", out)
        self._remove_padding(out)
        # print_dict("out", out)

        # check shape
        for k in ["REAL_verts"]:
            assert out[k] is None or out[k].shape == out["dfrm_verts"].shape

        ret = dict(animnet=out)
        ret["FACE_VIDX"] = FACE_VIDX
        ret["NON_FACE_VIDX"] = NON_FACE_VIDX
        return ret

    def _remove_padding(self, data_dict):
        if self.seq_info.has_padding:
            for k in data_dict:
                if isinstance(data_dict[k], dict):
                    self._remove_padding(data_dict[k])
                elif torch.is_tensor(data_dict[k]) and data_dict[k].shape[1] == self.seq_info.n_frames_padded:
                    data_dict[k] = self.seq_info.remove_padding(data_dict[k])
        return data_dict

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                  Optimizers                                                  * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def configure_optimizers(self):
        assert self.hparams.freeze_renderer
        assert not self.hparams.freeze_animnet
        # all parameters of animnet
        all_params = set(self.animnet.parameters())
        optim, lrsch = get_optim_and_lrsch(self.hparams.optim_animnet, list(all_params))
        return optim if lrsch is None else ([optim], [lrsch])

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                 Training part                                                * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def training_step(self, batch, batch_idx):
        # make sure train mode
        self.train()

        results = self.run_animnet(batch)
        loss, loss_logs = self.compute_animnet_loss(batch, results, True)
        metric_logs = self.get_metrics(batch, results, prefix="metric")

        # tensorboard log scalars
        logs = {**loss_logs, **metric_logs}
        self.log_dict(logs, prog_bar=False)
        # debug print
        if self.hparams.debug and self.global_step % 20 == 0:
            # print_dict("batch", batch)
            print_dict("loss", loss_logs, level="DEBUG")
            print_dict("metric", metric_logs, level="DEBUG")

        # tensorboard add images
        gap = self.hparams.visualizer.draw_gap_steps if not self.hparams.debug else 20
        if gap > 0 and self.global_step % gap == 0:
            n_plot = min(2, self.get_data_for_animnet(batch)["idle_verts"].shape[0])
            self.tb_add_images("train", n_plot, batch, results)

        return dict(loss=loss, items_log=logs)

    def validation_step(self, batch, batch_idx):
        # make sure eval mode
        self.eval()

        results = self(batch)
        loss, loss_logs = self.compute_animnet_loss(batch, results, False)
        metric_logs = self.get_metrics(batch, results, prefix="val_metric")

        logs = {**loss_logs, **metric_logs}
        self.log_dict(logs)

        return dict(val_loss=loss, items_log=logs)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                Compute Losses                                                * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def compute_animnet_loss(self, batch, results, training):
        dat = self.get_data_for_animnet(batch)

        if results["animnet"].get("imspc") is not None:
            ims = results["animnet"]["imspc"]
            code_dict = results["animnet"]["code_dict"]

            item_fake, logs_fake = self.loss_module.item_images(ims, dat, False, "nr_fake", "mask_neye", ":fake")
            item_tex3, logs_tex3 = self.loss_module.item_images(ims, dat, False, "nr_tex3", "mask_neye", ":tex3")
            item_lmks, logs_lmks = self.loss_module.item_landmarks(ims, dat)
            item_regs, logs_regs = self.loss_module.item_reg_codes(code_dict)
            item_rgsm, logs_rgsm = self.loss_module.item_reg_smooth(code_dict)

            loss = item_fake + item_tex3 * self.loss_module.cfg.extra_scale_tex3 + item_lmks + item_regs + item_rgsm
            logs = {**logs_fake, **logs_tex3, **logs_lmks, **logs_regs, **logs_rgsm}

            prefix = "loss/" if training else "val_loss/"
            logs = {prefix + k: v for k, v in logs.items()}
        else:
            pred = results["animnet"]["dfrm_verts"]
            REAL = (
                results["animnet"]["REAL_verts"]
                if results["animnet"].get("REAL_verts") is not None
                else results["animnet"]["trck_verts"]
            )
            loss_opts = self.hparams.loss.mesh
            ldict, wdict = compute_mesh_loss(loss_opts, pred, REAL, loss_opts.part)
            loss, logs = sum_up_losses(ldict, wdict, training)

        return loss, logs

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                    Metrics                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_metrics(self, batch, results, reduction="mean", prefix=None):
        dat = self.get_data_for_animnet(batch)
        res = results["animnet"]

        log = dict()

        # image metrics
        if dat.get("image") is not None and dotdict.fetch(res, "imspc.nr_fake") is not None:
            real = may_CHW2HWC(dat["image"])
            fake = may_CHW2HWC(res["imspc"]["nr_fake"])
            real_mask = may_CHW2HWC(dat["face_mask"])
            fake_mask = may_CHW2HWC(res["imspc"]["mask_neye"])

            real = torch.where(real_mask, real, torch.zeros_like(real))
            mask = torch.logical_or(real_mask, fake_mask)
            log["img:psnr(masked)"] = metrics.psnr_masked(fake, real, mask)
            log["img:ssim(masked)"] = metrics.ssim_masked(fake, real, mask)

        # lmks metrics
        if dat.get("lmks_fw75") is not None and dotdict.fetch(res, "imspc.lmks_fw75") is not None:
            real = dat["lmks_fw75"]
            fake = res["imspc"]["lmks_fw75"]
            # fmt: off
            log["lmd-avg:lip"] = metrics.lmd_lip  (fake, real, "mean")
            log["lmd-avg:inn"] = metrics.lmd_inner(fake, real, "mean")
            log["lmd-max:lip"] = metrics.lmd_lip  (fake, real, "max")
            log["lmd-max:inn"] = metrics.lmd_inner(fake, real, "max")
            # fmt: on

        # mesh metrics
        # REAL is feteched from results, which are copied and aligned from batch
        # fmt: off
        REAL = None
        if REAL is None: REAL = res.get("REAL_verts")
        if REAL is None: REAL = res.get("trck_verts")
        if REAL is not None:
            pred = res["dfrm_verts"]
            log["mvd-avg:lips" ] = metrics.verts_dist(pred, REAL, LIPS_VIDX, reduction='mean')
            log["mvd-max:lips" ] = metrics.verts_dist(pred, REAL, LIPS_VIDX, reduction='max')
            log["mvd-avg:lower"] = metrics.verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='mean')
            log["mvd-max:lower"] = metrics.verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='max')
            log["mvd-avg:face" ] = metrics.verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='mean')
            log["mvd-max:face" ] = metrics.verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='max')
        # fmt: on

        # reduce each metric and detach
        log = reduce_dict(log, reduction=reduction, detach=True)

        prefix = "" if prefix is None else (prefix + "/")
        return {prefix + k: v.detach() for k, v in log.items()}


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                               Pre-compute Paddings                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #


""" Run by '_init_' in config.model.  It will compute all padding information for sequential modules.  """


def pre_compute_paddings(config: DictConfig):
    config = config.animnet

    if config.sequential.using == "conv":
        from ..modules.seq_conv import SeqConv

        SeqConv.compute_padding_information(config)
    else:
        raise NotImplementedError()
