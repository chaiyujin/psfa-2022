from copy import deepcopy

import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.engine import ops
from src.engine.builder import load_or_build
from src.engine.logging import get_logger
from src.engine.train import get_optim_and_lrsch, print_dict, reduce_dict, sum_up_losses
from src.mm_fitting.libmorph import FLAME
from src.mm_fitting.libmorph.config import FLAMEConfig
from src.modules.neural_renderer import NeuralRenderer
from src.projects.anim import PL_AnimBase, SeqInfo
from src.projects.anim.utils_vis import generate_videos

from ..loss import compute_label_ce, compute_mesh_loss
from ..pl_animnet.forward import fw_renderer
from .animnet_mixer import AnimNetMixer

logger = get_logger(__name__)


class PL_AnimNetMixer(PL_AnimBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # * Neural Renderer: for Visualization and Image-Space Loss
        self.renderer = None
        if self.hparams.get("neural_renderer") is not None:
            self.renderer = load_or_build(self.hparams.neural_renderer, NeuralRenderer)
            # maybe frozen
            self.freeze_renderer = self.hparams.freeze_renderer
            if self.freeze_renderer:
                ops.freeze(self.renderer)

        # * AnimNet: Audio -> 3D Animation (Coefficients or Vertex-Offsets)
        self.animnet_mixer = load_or_build(self.hparams.animnet, AnimNetMixer)

        # * Flame
        self.flame = FLAME(FLAMEConfig())

        # * seq info
        self.seq_info = SeqInfo(self.hparams.animnet.src_seq_frames, self.hparams.animnet.src_seq_pads)

        self.using_label = self.hparams.using_label

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                          Specific Methods for Module                                         * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def final_offsets(self, results):
        return results["animnet"]["dfrm_verts"] - results["animnet"]["idle_verts"]

    def final_nr_fake(self, results):
        if results["animnet"].get("imspc") is not None:
            return results["animnet"]["imspc"].get("nr_fake")
        else:
            return None

    @torch.no_grad()
    def on_visualization(self, batch, results):
        if results.get("animnet") is None:
            return results

        dat = self.get_data_for_animnet(batch)
        res = results["animnet"]
        for key in ["dfrm_verts", "trck_verts"]:
            imspc_key = f"imspc-{key[:4]}" if key != "dfrm_verts" else "imspc"
            if res.get(imspc_key) is None and res.get(key) is not None and dat.get("track_data") is not None:
                res[imspc_key] = fw_renderer(self.renderer, res[key].detach(), dat, True)

        # logits -> prob -> label
        res["vis_label_ids_real"] = dat.get(f"{self.using_label}_ids")
        if res.get("logits_ctt_audio") is not None:
            prob = torch.softmax(res["logits_ctt_audio"], dim=-1)
            _ids = torch_dist.Categorical(probs=prob).sample()
            res["vis_label_prob_audio"] = prob
            res["vis_label_ids_audio"] = _ids
        if res.get("logits_ctt_refer") is not None:
            prob = torch.softmax(res["logits_ctt_refer"], dim=-1)
            _ids = torch_dist.Categorical(probs=prob).sample()
            res["vis_label_prob_refer"] = prob
            res["vis_label_ids_refer"] = _ids

        return results

    # The batch may contain mulitple dataset
    def get_data_for_animnet(self, batch):
        if "voca" in batch:
            return batch["voca"]
        return batch

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                  Optimizers                                                  * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def configure_optimizers(self):
        assert self.hparams.freeze_renderer
        assert not self.hparams.freeze_animnet, "'freeze_animnet' should not be True!"
        params = list(self.animnet_mixer.parameters())
        optim, lrsch = get_optim_and_lrsch(self.hparams.optim_animnet, params)
        return optim if lrsch is None else ([optim], [lrsch])

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                    AnimNet                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @torch.no_grad()
    def forward(self, batch, **kwargs):
        self.eval()
        return self.run_animnet(batch, **kwargs)

    def run_animnet(self, batch, **kwargs):
        # * forward
        out = self.animnet_mixer(**batch, **kwargs)

        # * set results dict
        idle_verts = batch["idle_verts"].unsqueeze(1)
        TRCK_delta = batch.get("offsets_tracked")
        REFR_delta = batch.get("offsets_refer")
        REAL_delta = batch.get("offsets_REAL")
        ret = dict(
            idle_verts=idle_verts,
            TRCK_verts=(idle_verts + TRCK_delta) if TRCK_delta is not None else None,
            REFR_verts=(idle_verts + REFR_delta) if REFR_delta is not None else None,
            REAL_verts=(idle_verts + REAL_delta) if REAL_delta is not None else None,
            dfrm_verts=out["out_verts_audio"],
            dfrm_verts_refer=out["out_verts_refer"],
            dfrm_verts_audio_nostyle=out["out_verts_audio_nostyle"],
            dfrm_verts_refer_nostyle=out["out_verts_refer_nostyle"],
            logits_ctt_audio=self.seq_info.remove_padding(out["logits_ctt_audio"]),
            logits_ctt_refer=self.seq_info.remove_padding(out["logits_ctt_refer"]),
            latent_ctt_audio=self.seq_info.remove_padding(out["latent_ctt_audio"]),
            latent_ctt_refer=self.seq_info.remove_padding(out["latent_ctt_refer"]),
        )

        # check shape
        for k in ["TRCK_verts", "REFR_verts", "REAL_verts"]:
            assert ret[k] is None or ret[k].shape == ret["dfrm_verts"].shape

        return dict(animnet=ret)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Training                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def random_shift_refer(self, batch):
        new_batch = {k: v for k, v in batch.items()}

        frm = batch["offsets_REAL"].shape[1]
        device = batch["offsets_REAL"].device
        idx = int(torch.randint(0, frm, (1,))[0])
        indices = torch.full((frm,), fill_value=idx, device=device)
        new_batch["offsets_refer"] = batch["offsets_refer"][:, indices]

        # new_batch["offsets_refer"] = batch["offsets_refer_mismatch"]

        return new_batch

    def training_step(self, batch, batch_idx):
        # make sure train mode
        self.train()

        # get loss and logs
        loss, logs = 0, dict()

        # loss for matched data pair
        dat = self.get_data_for_animnet(batch)
        results = self.run_animnet(dat)
        item0, logs0 = self.compute_loss(dat, results, training=True, audio=True, refer=True, prefix="")
        loss = loss + item0 * self.hparams.loss.scale.paired
        logs.update(logs0)

        # loss for mismatched refer lower
        mm_bat = self.random_shift_refer(dat)
        mm_res = self.run_animnet(mm_bat)
        item1, logs1 = self.compute_loss(mm_bat, mm_res, training=True, audio=True, refer=False, prefix="mismatch-")
        loss = loss + item1 * self.hparams.loss.scale.mismatch
        logs.update(logs1)

        # for coma
        if "coma" in batch and self.hparams.loss.scale.coma > 0:
            ds = dat["audio_dict"]["deepspeech"]
            coma_dat = batch["coma"]
            N, L = coma_dat["offsets_REAL"].shape[:2]
            shape = list(ds.shape)
            shape[0] = N
            shape[1] = L
            coma_dat["audio_dict"] = dict(deepspeech=torch.zeros(shape, dtype=ds.dtype, device=ds.device))
            coma_res = self.run_animnet(coma_dat)
            item2, logs2 = self.compute_loss(coma_dat, coma_res, training=True, audio=False, refer=True, prefix="coma-")
            loss = loss + item2 * self.hparams.loss.scale.coma
            logs.update(logs2)

        if self.hparams.loss.scale.random_flame > 0:
            flame_dat = deepcopy(dat)
            N, L = flame_dat["offsets_REAL"].shape[:2]
            dtype = flame_dat["offsets_REAL"].dtype
            device = flame_dat["offsets_REAL"].device
            exp = torch.randn((N * L, 50), dtype=dtype, device=device) * 0.7
            jaw = torch.rand((N * L, 3), dtype=dtype, device=device) * 0.3
            jaw.data[..., 1:].zero_()
            offsets = self.flame({"exp": exp, "jaw_pose": jaw}) - self.flame.v_template.unsqueeze(0)
            flame_dat["offsets_REAL"] = offsets.view(N, L, -1, 3)
            flame_dat["offsets_refer"] = offsets.view(N, L, -1, 3)
            flame_res = self.run_animnet(flame_dat)
            item3, logs3 = self.compute_loss(
                flame_dat, flame_res, training=True, audio=False, refer=True, prefix="flame-"
            )
            loss = loss + item3 * self.hparams.loss.scale.random_flame
            logs.update(logs3)

        # metrics
        logs.update(self.get_metrics(dat, results, prefix="metric"))

        # tensorboard log scalars
        self.log_dict(logs, prog_bar=False)
        # debug print
        if self.hparams.debug and self.global_step % 20 == 0:
            # print_dict("batch", batch)
            print_dict("logs", logs, level="DEBUG")

        # tensorboard add images
        gap = self.hparams.visualizer.draw_gap_steps if not self.hparams.debug else 20
        if gap > 0 and self.global_step % gap == 0:
            n_plot = min(2, dat["idle_verts"].shape[0])
            self.tb_add_images("train", n_plot, dat, results)
            self.tb_add_images("train", n_plot, mm_bat, mm_res, postfix="-mismatch")
            if "coma" in batch and self.hparams.loss.scale.coma > 0:
                self.tb_add_images("train", n_plot, coma_dat, coma_res, postfix="-coma")
            if self.hparams.loss.scale.random_flame > 0:
                self.tb_add_images("train", n_plot, flame_dat, flame_res, postfix="-flame")

        return dict(loss=loss, items_log=logs)

    def compute_loss(self, batch, results, training, audio, refer, prefix=""):
        dat = batch
        res = results["animnet"]

        ret_ldict, ret_wdict = dict(), dict()
        loss_opts = self.hparams.loss

        if audio:
            ldict, wdict = compute_mesh_loss(loss_opts.mesh, res["dfrm_verts"], res["REAL_verts"], "dfrm")
            ret_ldict.update(ldict)
            ret_wdict.update(wdict)

            if audio and refer and dat.get(f"{self.using_label}_ids") is not None:
                ldict, wdict = compute_label_ce(
                    loss_opts=loss_opts.label,
                    pred_logits=res["logits_ctt_audio"],
                    real_ids=dat[f"{self.using_label}_ids"],
                    using_label=self.using_label,
                    tag="audio",
                )
                ret_ldict.update(ldict)
                ret_wdict.update(wdict)

        if refer:
            ldict, wdict = compute_mesh_loss(loss_opts.mesh, res["dfrm_verts_refer"], res["REAL_verts"], "dfrm_refer")
            ret_ldict.update(ldict)
            ret_wdict.update(wdict)

            if audio and refer and dat.get(f"{self.using_label}_ids") is not None:
                ldict, wdict = compute_label_ce(
                    loss_opts=loss_opts.label,
                    pred_logits=res["logits_ctt_refer"],
                    real_ids=dat[f"{self.using_label}_ids"],
                    using_label=self.using_label,
                    tag="refxy",
                )
                ret_ldict.update(ldict)
                ret_wdict.update(wdict)

        if audio and refer:
            using = self.hparams.loss.latent.using
            item = self._latent_dist(res["latent_ctt_audio"], res["latent_ctt_refer"], using)
            ret_ldict[f"latent:audio-refer({using})"] = item
            ret_wdict[f"latent:audio-refer({using})"] = loss_opts.latent.paired_close

        return sum_up_losses(ret_ldict, ret_wdict, training=training, extra_prefix=prefix)

    def _latent_dist(self, z0, z1, using):
        if using == "l2":
            item = ((z0 - z1) ** 2).mean(-1) * 0.5
        elif using == "cos":
            item = 1.0 - F.cosine_similarity(z0, z1, dim=-1)
        else:
            raise NotImplementedError("Unknown 'loss.latent.using': '{}'".format(using))
        return item

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Generate                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def generate(self, epoch, global_step, generate_dir, during_training=False, datamodule=None):
        self._generating = True
        generate_videos(self, epoch, generate_dir, during_training, datamodule, generator_class=self._generator_cls)
        self._generating = False

    def get_metrics(self, batch, results, reduction="mean", prefix=None):
        return {}


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                               Pre-compute Paddings                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

"""
This function is called before all classes' __init__, set by '_init_' in experiment yaml.
It will compute all padding information for sequential modules.
Arguments:
    top_config: given config is at top level.
"""


def pre_compute_paddings(top_config: DictConfig):
    return
    from omegaconf import open_dict

    from src.utils.misc.ascii import Table

    config = top_config.animnet
    with open_dict(config):
        config.src_seq_pads = [1, 1]

    # fmt: off
    table = Table(alignment=("left", "middle"))
    table.add_row("animnet.src_seq_frames", config.src_seq_frames)
    table.add_row("animnet.tgt_seq_frames", config.tgt_seq_frames)
    table.add_row("animnet.src_seq_pads",   str(config.src_seq_pads))
    logger.info("Pre-Computed Paddings:\n{}".format(table))
