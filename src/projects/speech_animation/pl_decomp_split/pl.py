from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src import constants, metrics
from src.data.assets import FACE_LOWER_VIDX, FACE_NEYE_VIDX, FACE_NOEYEBALLS_VIDX, LIPS_VIDX
from src.datasets.base.anim import AnimBaseDataset
from src.engine import ops
from src.engine.builder import load_or_build
from src.engine.logging import get_logger
from src.engine.train import get_optim_and_lrsch, get_trainset, get_validset, print_dict, reduce_dict, sum_up_losses
from src.modules.neural_renderer import NeuralRenderer
from src.projects.anim import PL_AnimBase, SeqInfo, VideoGenerator, get_painters

from ..loss import compute_mesh_loss
from . import visualize as _
from .model import AnimNetDecmp

log = get_logger("AnimNet")

FACE_VIDX = FACE_NOEYEBALLS_VIDX
NON_FACE_VIDX = [x for x in range(5023) if x not in FACE_VIDX]


class PL_AnimNetDecmp(PL_AnimBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # * Neural Renderer: for Visualization and Image-Space Loss
        self.renderer: Optional[nn.Module] = None
        if self.hparams.get("neural_renderer") is not None:
            self.renderer = load_or_build(self.hparams.neural_renderer, NeuralRenderer)
            # maybe frozen
            self.freeze_renderer = self.hparams.freeze_renderer
            if self.freeze_renderer:
                ops.freeze(self.renderer)

        # * AnimNet: Audio -> 3D Animation (Coefficients or Vertex-Offsets)
        m: Optional[AnimNetDecmp] = load_or_build(self.hparams.animnet, AnimNetDecmp)
        assert m is not None
        self.animnet_decmp: AnimNetDecmp = m
        # freeze flag
        self.freeze_animnet = self.hparams.freeze_animnet
        assert not self.freeze_animnet

        # * Seq info
        self.seq_info = SeqInfo(self.hparams.animnet.src_seq_frames, self.hparams.animnet.src_seq_pads)

        # Flag to control generation of debug results
        self.tb_vis = False

        # load from pre-trained model
        cfg = self.hparams
        if cfg.get("load_from") is not None:
            state_dict = torch.load(cfg.load_from, map_location="cpu")
            self.load_state_dict(state_dict["state_dict"])
            log.info("Load pre-trained model from {}".format(cfg.load_from))

        # freeze or not (after pre-training)
        self.freeze_audio_encoder = cfg.get("freeze_audio_encoder", False)
        self.freeze_decoder = cfg.get("freeze_decoder", False)
        if self.freeze_audio_encoder:
            ops.freeze(self.animnet_decmp.enc_ctt_aud)
            ops.freeze(self.animnet_decmp.enc_upp_aud)
            log.info("Freeze audio_encoders")
        if self.freeze_decoder:
            ops.freeze(self.animnet_decmp.decoder_verts)
            log.info("Freeze verts_decoder")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.animnet_decmp.state_dict(destination, prefix + "animnet_decmp.", keep_vars)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                          Specific Methods for Module                                         * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def to_dump_offsets(self, results, fi) -> Optional[Dict[str, Any]]:
        return {"final": results["pred"][0, fi]}

    @torch.no_grad()
    def forward(self, batch, **kwargs):
        self.eval()

        idle_verts = batch["idle_verts"]
        X = batch["audio_dict"]

        Y_IP = None
        if batch.get("offsets_REAL") is not None:
            Y_IP = batch.get("offsets_REAL")
        elif batch.get("offsets_tracked") is not None:
            assert self.generating
            Y_IP = batch.get("offsets_tracked")

        assert batch.get("offsets_swap_style") is not None
        Y_JQ = batch["offsets_swap_style"]  # aligned with Y_IP, thus same content

        out = dict()
        # * Audio + Style
        z_ctt, z_upp = self.animnet_decmp.encode_audio(X)
        _, _, z_sty = self.animnet_decmp.decomp_offsets(Y_JQ)
        out["pred"] = self.animnet_decmp.remix(z_ctt, z_upp, z_sty, idle_verts)["out_verts"] - idle_verts.unsqueeze(1)

        if Y_IP is not None:
            out["Y"] = Y_IP
            out["Y_IP"], out["Y_JQ"] = Y_IP, Y_JQ
            out["Y_IQ"], out["Y_JP"] = Y_JQ, Y_IP
            out["X_I"], out["X_J"] = X, X
            out.update(self.duplex(idle_verts, **out))

        out["FACE_VIDX"] = FACE_LOWER_VIDX
        out["NON_FACE_VIDX"] = NON_FACE_VIDX
        self._remove_padding(out)
        return out

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                  Optimizers                                                  * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def configure_optimizers(self):
        assert self.hparams.freeze_renderer
        assert not self.hparams.freeze_animnet
        # all parameters of animnet
        all_params = set(self.animnet_decmp.parameters())
        optim, lrsch = get_optim_and_lrsch(self.hparams.optim_animnet, list(all_params))
        return optim if lrsch is None else ([optim], [lrsch])

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                 Training part                                                * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_Y(self, dat):
        Y = None
        for key in ["offsets_REAL", "offsets_tracked"]:
            if key in dat:
                Y = dat[key]
        return Y

    def duplex(self, idle_verts, **data):
        # fmt: off
        def _enc_audio(out, c, X):
            out[f"z_ctt_aud_{c}"], out[f"z_upp_aud_{c}"] = self.animnet_decmp.encode_audio(X)

        def _enc_anime(out, c, s, Y):
            out[f"z_ctt_ani_{c}"], out[f"z_upp_ani_{c}"], out[f"z_sty_{s}"] = self.animnet_decmp.decomp_offsets(Y)

        def _remix(z_ctt, z_upp, z_sty, set_zeros=()):
            if 'z_ctt' in set_zeros: z_ctt = torch.zeros_like(z_ctt)
            if 'z_upp' in set_zeros: z_upp = torch.zeros_like(z_upp)
            if 'z_sty' in set_zeros: z_sty = torch.zeros_like(z_sty)
            return self.animnet_decmp.remix(z_ctt, z_upp, z_sty, idle_verts)["out_verts"] - idle_verts.unsqueeze(1)

        # First Phase
        ph1 = dict()
        for key in ["ip", "jq"]:
            c, s = key[0], key[1]
            # encode
            _enc_audio(ph1, c, data[f"X_{c.upper()}"])
            _enc_anime(ph1, c, s, data[f"Y_{key.upper()}"])
            # reconstruct with audio / content
            ph1[f"y_{key}_rec_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_upp_ani_{c}"], ph1[f"z_sty_{s}"])
            ph1[f"y_{key}_rec_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_upp_aud_{c}"], ph1[f"z_sty_{s}"])
            # reconstruct with zero upp only
            ph1[f"y_{key}_rec_low_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_upp_ani_{c}"], ph1[f"z_sty_{s}"], ["z_upp"])
            ph1[f"y_{key}_rec_low_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_upp_aud_{c}"], ph1[f"z_sty_{s}"], ["z_upp"])
            # debug: zero content or zero style
            if self.tb_vis or self.generating:
                ph1[f"y_0{s}_vis_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_upp_ani_{c}"], ph1[f"z_sty_{s}"], ["z_ctt", "z_upp"])
                ph1[f"y_{c}0_vis_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_upp_ani_{c}"], ph1[f"z_sty_{s}"], ["z_sty"])
                ph1[f"y_{c}0_vis_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_upp_aud_{c}"], ph1[f"z_sty_{s}"], ["z_sty"])
        # swap
        for key in ["iq", "jp"]:
            c, s = key[0], key[1]
            ph1[f"y_{key}_swp_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_upp_ani_{c}"], ph1[f"z_sty_{s}"])
            ph1[f"y_{key}_swp_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_upp_aud_{c}"], ph1[f"z_sty_{s}"])

        # * Second Phase
        ph2 = dict()
        for key in ["iq", "jp"]:
            c, s = key[0], key[1]
            # copy audio latents from phase_1st
            ph2[f"z_ctt_aud_{c}"] = ph1[f"z_ctt_aud_{c}"]
            ph2[f"z_upp_aud_{c}"] = ph1[f"z_upp_aud_{c}"]
            # encode result from phase_1st
            y_key = ph1[f"y_{key}_swp_ani"]
            Y_KEY = data[f"Y_{key.upper()}"]
            # WARN: Use real data to pad results from first phase
            if self.seq_info.has_padding:
                assert self.seq_info.n_frames_padded == Y_KEY.shape[1]
                padl, padr = self.seq_info.padding
                # fmt: off
                if padl > 0: y_key = torch.cat((Y_KEY[:, :padl], y_key), dim=1)
                if padr > 0: y_key = torch.cat((y_key, Y_KEY[:, -padr:]), dim=1)
                # fmt: on
            _enc_anime(ph2, c, s, y_key)
            # debug: zero content or zero style
            if self.tb_vis or self.generating:
                ph2[f"y_0{s}_vis_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_upp_ani_{c}"], ph2[f"z_sty_{s}"], ["z_ctt", "z_upp"])
                ph2[f"y_{c}0_vis_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_upp_ani_{c}"], ph2[f"z_sty_{s}"], ["z_sty"])
                ph2[f"y_{c}0_vis_aud"] = _remix(ph2[f"z_ctt_aud_{c}"], ph2[f"z_upp_aud_{c}"], ph2[f"z_sty_{s}"], ["z_sty"])
        # swap
        for key in ["ip", "jq"]:
            c, s = key[0], key[1]
            ph2[f"y_{key}_cyc_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_upp_ani_{c}"], ph2[f"z_sty_{s}"])
            ph2[f"y_{key}_cyc_aud"] = _remix(ph2[f"z_ctt_aud_{c}"], ph2[f"z_upp_aud_{c}"], ph2[f"z_sty_{s}"])

        out = dict()
        out["phase_1st"] = ph1
        out["phase_2nd"] = ph2
        # copy vidx
        out["FACE_VIDX"] = FACE_VIDX
        out["NON_FACE_VIDX"] = NON_FACE_VIDX
        # fmt: on
        return out

    def _remove_padding(self, data_dict):
        if self.seq_info.has_padding:
            for k in data_dict:
                if isinstance(data_dict[k], dict):
                    self._remove_padding(data_dict[k])
                elif torch.is_tensor(data_dict[k]) and data_dict[k].shape[1] == self.seq_info.n_frames_padded:
                    data_dict[k] = self.seq_info.remove_padding(data_dict[k])
        return data_dict

    def training_step(self, batch, batch_idx):
        # make sure train mode
        self.train()
        gap = self.hparams.visualizer.draw_gap_steps if not self.hparams.debug else 20
        self.tb_vis = gap > 0 and self.global_step % gap == 0

        # get data
        dat = batch
        idle_verts = dat["I"]["idle_verts"]
        batch["idle_verts"] = idle_verts

        # output
        out = dict()
        out["Y_IP"], out["Y_IQ"] = self.get_Y(dat["I"]), dat["I"].get("offsets_swap_style")
        out["Y_JQ"], out["Y_JP"] = self.get_Y(dat["J"]), dat["J"].get("offsets_swap_style")
        out["X_I"] = dat["I"].get("audio_dict")
        out["X_J"] = dat["J"].get("audio_dict")
        out.update(self.duplex(idle_verts, **out))
        # remove the paddings (optional)
        self._remove_padding(out)

        # loss
        loss, logs = self.get_loss(out, training=True)

        self.log_dict(logs)
        if self.hparams.debug and self.global_step % 10 == 0:
            print_dict("logs", logs)

        # tensorboard add images
        if self.tb_vis:
            self.tb_add_images("train", 1, dat, out)

        return dict(loss=loss, items_log=logs)

    def validation_step(self, batch, batch_idx):

        # make sure eval mode
        self.eval()
        self.tb_vis = False

        out = self(batch)
        loss, logs = 0, self.get_metrics(batch, out, prefix="val_metric")
        # print_dict("metric", logs)

        return dict(val_loss=loss, items_log=logs)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                     Loss                                                     * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_loss(self, out, training):
        loss_opts = self.hparams.loss
        ldict, wdict = dict(), dict()

        def _loss_fn(tag, inp, tar, scale, part, **kwargs):
            ls, ws = compute_mesh_loss(self.hparams.loss.mesh, inp, tar, part, **kwargs, key_fmt=tag + "-{}")
            for key in ls:
                if key not in ldict:
                    ldict[key] = ls[key]
                    wdict[key] = ws[key] * scale
                else:
                    assert wdict[key] == ws[key] * scale
                    ldict[key] = ldict[key] + ls[key]

        # fmt: off

        # INFO: Reconstuction loss item, should reconstruct the target part (from config)
        if loss_opts.rec > 0:
            part = self.hparams.loss.mesh.part
            scale = loss_opts.rec * 0.5
            _loss_fn("ani:rec", out["phase_1st"]["y_ip_rec_ani"], out["Y_IP"], scale, part)
            _loss_fn("ani:rec", out["phase_1st"]["y_jq_rec_ani"], out["Y_JQ"], scale, part)
            scale = loss_opts.rec * 0.5 * loss_opts.audio
            _loss_fn("aud:rec", out["phase_1st"]["y_ip_rec_aud"], out["Y_IP"], scale, part)
            _loss_fn("aud:rec", out["phase_1st"]["y_jq_rec_aud"], out["Y_JQ"], scale, part)

        # WARN: Loss for 'low'
        if loss_opts.rec > 0:
            part_low = "face_below_eye"
            part_upp = "eyebrows"
            exkw_low = dict(scale_inv=0.0, scale_oth=0.0)
            exkw_upp = dict(scale_inv=0.0, scale_oth=0.0, scale_lip=0.0, want_mean_pos=True, want_zero_mot=True)
            scale = loss_opts.rec * 0.5
            _loss_fn("ani:rec_low", out["phase_1st"]["y_ip_rec_low_ani"], out["Y_IP"], scale, part_low, **exkw_low)
            _loss_fn("ani:rec_low", out["phase_1st"]["y_jq_rec_low_ani"], out["Y_JQ"], scale, part_low, **exkw_low)
            _loss_fn("ani:reg_upp", out["phase_1st"]["y_ip_rec_low_ani"], out["Y_IP"], scale, part_upp, **exkw_upp)
            _loss_fn("ani:reg_upp", out["phase_1st"]["y_jq_rec_low_ani"], out["Y_JQ"], scale, part_upp, **exkw_upp)
            scale = loss_opts.rec * 0.5 * loss_opts.audio
            _loss_fn("aud:rec_low", out["phase_1st"]["y_ip_rec_low_aud"], out["Y_IP"], scale, part_low, **exkw_low)
            _loss_fn("aud:rec_low", out["phase_1st"]["y_jq_rec_low_aud"], out["Y_JQ"], scale, part_low, **exkw_low)
            _loss_fn("aud:reg_upp", out["phase_1st"]["y_ip_rec_low_aud"], out["Y_IP"], scale, part_upp, **exkw_upp)
            _loss_fn("aud:reg_upp", out["phase_1st"]["y_jq_rec_low_aud"], out["Y_JQ"], scale, part_upp, **exkw_upp)

        # INFO: The Loss for swap (1st phase), only supervise lower face part
        if loss_opts.swp > 0:
            # part = self.hparams.loss.mesh.part
            part = "face_below_eye"
            exkw = dict(scale_inv=0.0, scale_oth=0.0)
            scale = loss_opts.swp * 0.5
            _loss_fn("ani:swp", out["phase_1st"]["y_iq_swp_ani"], out["Y_IQ"], scale, part, **exkw)
            _loss_fn("ani:swp", out["phase_1st"]["y_jp_swp_ani"], out["Y_JP"], scale, part, **exkw)
            scale = loss_opts.swp * 0.5 * loss_opts.audio
            _loss_fn("aud:swp", out["phase_1st"]["y_iq_swp_aud"], out["Y_IQ"], scale, part, **exkw)
            _loss_fn("aud:swp", out["phase_1st"]["y_jp_swp_aud"], out["Y_JP"], scale, part, **exkw)

        # INFO: The Loss for cyc (2nd phase), should reconstruct the target part (from config)
        if loss_opts.cyc > 0:
            part = self.hparams.loss.mesh.part
            scale = loss_opts.cyc * 0.5
            _loss_fn("ani:cyc", out["phase_2nd"]["y_ip_cyc_ani"], out["Y_IP"], scale, part)
            _loss_fn("ani:cyc", out["phase_2nd"]["y_jq_cyc_ani"], out["Y_JQ"], scale, part)
            scale = loss_opts.cyc * 0.5 * loss_opts.audio
            _loss_fn("aud:cyc", out["phase_2nd"]["y_ip_cyc_aud"], out["Y_IP"], scale, part)
            _loss_fn("aud:cyc", out["phase_2nd"]["y_jq_cyc_aud"], out["Y_JQ"], scale, part)

        # INFO: Regularization of latents
        def _l2_reg(a, b):
            return ((a - b) ** 2).mean() * 0.5

        if loss_opts.reg_z_ctt_close > 0:
            item_ctt_i = _l2_reg(out["phase_1st"]["z_ctt_ani_i"], out["phase_2nd"]["z_ctt_ani_i"])
            item_ctt_j = _l2_reg(out["phase_1st"]["z_ctt_ani_j"], out["phase_2nd"]["z_ctt_ani_j"])
            ldict["reg:z_ctt_close"] = item_ctt_i + item_ctt_j
            wdict["reg:z_ctt_close"] = loss_opts.reg_z_ctt_close

        if loss_opts.reg_z_upp_close > 0:
            item_upp_i = _l2_reg(out["phase_1st"]["z_upp_ani_i"], out["phase_2nd"]["z_upp_ani_i"])
            item_upp_j = _l2_reg(out["phase_1st"]["z_upp_ani_j"], out["phase_2nd"]["z_upp_ani_j"])
            ldict["reg:z_upp_close"] = item_upp_i + item_upp_j
            wdict["reg:z_upp_close"] = loss_opts.reg_z_upp_close

        if loss_opts.reg_z_sty_close > 0:
            item_sty_p = _l2_reg(out["phase_1st"]["z_sty_p"], out["phase_2nd"]["z_sty_p"])
            item_sty_q = _l2_reg(out["phase_1st"]["z_sty_q"], out["phase_2nd"]["z_sty_q"])
            ldict["reg:z_sty_close"] = item_sty_p + item_sty_q
            wdict["reg:z_sty_close"] = loss_opts.reg_z_sty_close

        if loss_opts.reg_z_aud_close > 0:  # and self.using_audio:
            wdict["reg:z_aud_close"] = loss_opts.reg_z_aud_close
            ldict["reg:z_aud_close"] = (
                _l2_reg(out["phase_1st"]["z_ctt_ani_i"], out["phase_1st"]["z_ctt_aud_i"]) +
                _l2_reg(out["phase_1st"]["z_ctt_ani_j"], out["phase_1st"]["z_ctt_aud_j"]) +
                _l2_reg(out["phase_2nd"]["z_ctt_ani_i"], out["phase_2nd"]["z_ctt_aud_i"]) +
                _l2_reg(out["phase_2nd"]["z_ctt_ani_j"], out["phase_2nd"]["z_ctt_aud_j"]) +
                _l2_reg(out["phase_1st"]["z_upp_ani_i"], out["phase_1st"]["z_upp_aud_i"]) +
                _l2_reg(out["phase_1st"]["z_upp_ani_j"], out["phase_1st"]["z_upp_aud_j"]) +
                _l2_reg(out["phase_2nd"]["z_upp_ani_i"], out["phase_2nd"]["z_upp_aud_i"]) +
                _l2_reg(out["phase_2nd"]["z_upp_ani_j"], out["phase_2nd"]["z_upp_aud_j"])
            )

        # fmt: on

        return sum_up_losses(ldict, wdict, training)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                    Metrics                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_metrics(self, batch, results, reduction="mean", prefix=None):
        res = results

        log = dict()

        # mesh metrics
        # REAL is feteched from results, which are copied and aligned from batch
        # fmt: off
        REAL = res.get("Y")
        pred = res.get("pred")
        if REAL is not None and pred is not None:
            log["mvd-avg:lips" ] = metrics.verts_dist(pred, REAL, LIPS_VIDX, reduction='mean')
            log["mvd-max:lips" ] = metrics.verts_dist(pred, REAL, LIPS_VIDX, reduction='max')
            log["mvd-avg:lower"] = metrics.verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='mean')
            log["mvd-max:lower"] = metrics.verts_dist(pred, REAL, FACE_LOWER_VIDX, reduction='max')
            log["mvd-avg:face" ] = metrics.verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='mean')
            log["mvd-max:face" ] = metrics.verts_dist(pred, REAL, FACE_NEYE_VIDX,  reduction='max')
        # fmt: on

        # reduce each metric and detach
        log = reduce_dict(log, reduction=reduction, detach=True)
        # print_dict("metric", log)

        prefix = "" if prefix is None else (prefix + "/")
        return {prefix + k: v.detach() for k, v in log.items()}

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Generate                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

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

        # * shared kwargs
        draw_fn_list = get_painters(self.hparams.visualizer)
        shared_kwargs: Dict[str, Any] = dict(
            epoch=epoch, media_dir=generate_dir, draw_fn_list=draw_fn_list, extra_tag=extra_tag
        )
        # generator
        generator = VideoGenerator(self)

        # * Test: get media list
        media_list = self.parse_media_list()
        if self.hparams.visualizer.generate_test:
            generator.generate_for_media(media_list=media_list, **shared_kwargs)

        # * Dataset: trainset or validset
        trainset = get_trainset(self, datamodule)
        validset = get_validset(self, datamodule)
        if isinstance(trainset, dict) and trainset.get("talk_video") is not None:
            trainset = trainset["talk_video"]
        elif isinstance(trainset, dict) and trainset.get("voca") is not None:
            trainset = trainset["voca"]

        def _generate_dataset():
            assert validset is None or isinstance(validset, AnimBaseDataset)
            assert trainset is None or isinstance(trainset, AnimBaseDataset)

            if self.hparams.visualizer.generate_valid and validset is not None:
                seq_list = self.hparams.visualizer.get("validset_seq_list") or validset_seq_list
                generator.generate_for_dataset(dataset=validset, set_type="valid", seq_list=seq_list, **shared_kwargs)

            if self.hparams.visualizer.generate_train and trainset is not None:
                seq_list = self.hparams.visualizer.get("trainset_seq_list") or trainset_seq_list
                generator.generate_for_dataset(dataset=trainset, set_type="train", seq_list=seq_list, **shared_kwargs)

        swp_spk_list = self.hparams.visualizer.get("swap_speakers", [None])
        for swp_spk in swp_spk_list:
            constants.KWARGS["swap_speaker"] = swp_spk
            constants.KWARGS["swap_speaker_range"] = "seq"
            swp_tag = f"swp_{swp_spk[16:]}-seq" if swp_spk is not None else None
            shared_kwargs["extra_tag"] = swp_tag
            _generate_dataset()
            constants.KWARGS["swap_speaker_range"] = "rnd_frm"
            swp_tag = f"swp_{swp_spk[16:]}-rnd_frm" if swp_spk is not None else None
            shared_kwargs["extra_tag"] = swp_tag
            _generate_dataset()
        constants.KWARGS.clear()

        constants.GENERATING = False
        self._generating = False

        # # * visualize the offsets by t-SNE
        # from .do_tsne import do_tsne

        # do_tsne(epoch, self.hparams.visualizer.speaker)


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
