from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict

from assets import FACE_LOWER_VIDX, FACE_NEYE_VIDX, FACE_NOEYEBALLS_VIDX, LIPS_VIDX
from src import constants, metrics
from src.datasets.base.anim import AnimBaseDataset
from src.engine.builder import load_or_build
from src.engine.logging import get_logger
from src.engine.train import get_optim_and_lrsch, get_trainset, get_validset, print_dict, reduce_dict, sum_up_losses
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

        log.info(f">>> {self.__class__.__name__}")

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

        # * Ablation flags
        self.ablation = self.hparams.ablation
        with_list = []
        for key, val in self.ablation.items():
            if val:
                with_list.append(key)
        # assert len(with_list) <= 1, "At most one ablation! But {}".format(with_list)
        if len(with_list) > 0:
            log.warn(">>> Ablation:[\n  {}\n]".format("\n  ".join(with_list)))
        if constants.MODE.lower() == "train":
            self._check_ablations()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.animnet_decmp.state_dict(destination, prefix + "animnet_decmp.", keep_vars)

    def _check_ablations(self):
        if self.ablation.e2e:
            assert self.hparams.loss.swp <= 0
            assert self.hparams.loss.cyc <= 0
            assert self.hparams.loss.reg_z_ctt_close <= 0
            assert self.hparams.loss.reg_z_upp_close <= 0
            assert self.hparams.loss.reg_z_sty_close <= 0
            assert self.hparams.loss.reg_z_aud_close <= 0
            assert not self.hparams.datamodule_sampler_kwargs.pairs_from_diff_speaker
            assert not self.hparams.datamodule_sampler_kwargs.pairs_from_same_speaker
        if self.ablation.no_dtw:
            # It's controlled by data sampler
            assert not self.hparams.datamodule_sampler_kwargs.pairs_from_diff_speaker
        if self.ablation.no_swp:
            assert self.hparams.loss.swp <= 0
        if self.ablation.no_ph2:
            assert self.hparams.loss.cyc <= 0
        if self.ablation.no_cyc:
            assert self.hparams.loss.cyc <= 0
        assert not (self.ablation.no_ph2 and self.ablation.no_cyc), "Cannot no_ph2 and no_cyc at same time!"
        if self.ablation.no_reg:
            assert self.hparams.loss.reg_z_ctt_close <= 0
            assert self.hparams.loss.reg_z_upp_close <= 0
            assert self.hparams.loss.reg_z_sty_close <= 0
            assert self.hparams.loss.reg_z_aud_close <= 0

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                          Specific Methods for Module                                         * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def to_dump_offsets(self, results, fi) -> Optional[Dict[str, Any]]:
        ret = {"final": results["pred"][0, fi]}
        if self.hparams.visualizer.get("dump_offsets_decomp", False):
            ret["only_sty"] = results["only_sty"][0, fi]
            ret["only_aud"] = results["only_aud"][0, fi]
            if "only_ctt" in results:
                ret["only_ctt"] = results["only_ctt"][0, fi]
            if "ctt_sty" in results:
                ret["ctt_sty"] = results["ctt_sty"][0, fi]
            if "input" in results:
                ret["input"] = results["input"][0, fi]
            ret["z_sty"] = results["z_sty"][0, fi]
            # HACK: save the reference sequence (Y_JQ)
            if "refer" in results and fi == 0:
                assert results["refer"].shape[0] == 1
                ret["refer"] = results["refer"][0]
                ret["refer_only_ctt"] = results["refer_only_ctt"][0]
                ret["refer_ctt_sty"] = results["refer_ctt_sty"][0]
            # print(ret["input"].shape, ret["z_sty"].shape, ret["z_sty"][:3])
        return ret

    @torch.no_grad()
    def forward(self, batch, **kwargs):
        self.eval()

        idle_verts = batch["idle_verts"]
        X = batch["audio_dict"]

        Y_IP = None
        Y_JQ = None
        if batch.get("offsets_REAL") is not None:
            Y_IP = batch.get("offsets_REAL")
        elif batch.get("offsets_tracked") is not None:
            assert self.generating
            Y_IP = batch.get("offsets_tracked")

        out = dict()
        if Y_IP is not None:
            out["Y"] = Y_IP

        # * Audio + Style
        z_ctt = None
        study_decoupling = self.hparams.visualizer.get("decoupling", False)

        if self.generating:
            # NOTE: if we are studying decoupling, we don't cache it
            if study_decoupling:
                Y_JQ = Y_IP
                z_ctt, z_sty = self.animnet_decmp.decomp_offsets(Y_JQ)
                if "offsets_swap_style" in batch:
                    del batch["offsets_swap_style"]
            else:
                if "z_style" not in kwargs["cache"]:
                    assert batch.get("offsets_swap_style") is not None
                    Y_JQ = batch["offsets_swap_style"]  # aligned with Y_IP, thus same content
                    _, z_sty = self.animnet_decmp.decomp_offsets(Y_JQ)
                    pl, pr = self.seq_info.padding
                    Y_JQ_padded = F.pad(Y_JQ, (0, 0, 0, 0, pl, pr))
                    z_ctt, _ = self.animnet_decmp.decomp_offsets(Y_JQ_padded)
                    # print(Y_JQ.shape, Y_JQ_padded.shape, self.seq_info.padding)
                    kwargs["cache"]["z_style"] = z_sty.detach()
                    del batch["offsets_swap_style"]
                    # print('get z_sty', z_sty.shape)
                else:
                    z_sty = kwargs["cache"]["z_style"]
                    # print('reuse z_sty')
        else:
            if self._valid_z_sty is None:
                assert batch.get("offsets_swap_style") is not None
                Y_JQ = batch["offsets_swap_style"]  # aligned with Y_IP, thus same content
                _, z_sty = self.animnet_decmp.decomp_offsets(Y_JQ)
                self._valid_z_sty = z_sty.clone()
                log.info("validation compute z_sty from src.data {}".format(Y_JQ.shape))
            else:
                # log.info("validaiton reuse z_sty")
                z_sty = self._valid_z_sty[: X["deepspeech"].shape[0]]

        z_aud = self.animnet_decmp.encode_audio(X)
        out["pred"] = self.animnet_decmp.remix(z_aud, z_sty, idle_verts)["out_verts"] - idle_verts.unsqueeze(1)

        if self.generating and self.hparams.visualizer.get("dump_offsets_decomp", False):
            if study_decoupling:
                out["input"] = Y_IP
            out["z_sty"] = z_sty.clone()
            # The 'only' animation
            out["only_sty"] = self.animnet_decmp.remix(torch.zeros_like(z_aud), z_sty, idle_verts)["out_verts"]
            out["only_aud"] = self.animnet_decmp.remix(z_aud, torch.zeros_like(z_sty), idle_verts)["out_verts"]
            out["only_sty"] -= idle_verts.unsqueeze(1)
            out["only_aud"] -= idle_verts.unsqueeze(1)
            if study_decoupling:
                assert z_ctt is not None
                assert z_ctt.shape == z_aud.shape
                out["only_ctt"] = self.animnet_decmp.remix(z_ctt, torch.zeros_like(z_sty), idle_verts)["out_verts"]
                out["ctt_sty"] = self.animnet_decmp.remix(z_ctt, z_sty, idle_verts)["out_verts"]
                out["only_ctt"] -= idle_verts.unsqueeze(1)
                out["ctt_sty"] -= idle_verts.unsqueeze(1)
            elif Y_JQ is not None:
                out["refer"] = Y_JQ
                assert z_ctt is not None
                out["refer_only_ctt"] = self.animnet_decmp.remix(z_ctt, torch.zeros_like(z_sty), idle_verts)[
                    "out_verts"
                ]
                out["refer_ctt_sty"] = self.animnet_decmp.remix(z_ctt, z_sty, idle_verts)["out_verts"]
                out["refer_only_ctt"] -= idle_verts.unsqueeze(1)
                out["refer_ctt_sty"] -= idle_verts.unsqueeze(1)

        # if Y_IP is not None:
        #     out["Y_IP"], out["Y_JQ"] = Y_IP, Y_JQ
        #     out["Y_IQ"], out["Y_JP"] = Y_JQ, Y_IP
        #     out["X_I"], out["X_J"] = X, X
        #     out.update(self.duplex(idle_verts, **out))

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
        # print(Y.shape)
        return Y

    def duplex(self, idle_verts, **data):
        # fmt: off
        def _enc_audio(res, c, X):
            res[f"z_ctt_aud_{c}"] = self.animnet_decmp.encode_audio(X)

        def _enc_anime(res, c, s, Y):
            res[f"z_ctt_ani_{c}"], res[f"z_sty_{s}"] = self.animnet_decmp.decomp_offsets(Y)

        def _remix(z_ctt, z_sty, set_zeros=()):
            if 'z_ctt' in set_zeros: z_ctt = torch.zeros_like(z_ctt)
            if 'z_sty' in set_zeros: z_sty = torch.zeros_like(z_sty)
            return self.animnet_decmp.remix(z_ctt, z_sty, idle_verts)["out_verts"] - idle_verts.unsqueeze(1)

        out = dict()

        # First Phase
        ph1 = dict()
        for key in ["ip", "jq"]:
            c, s = key[0], key[1]
            # encode
            _enc_audio(ph1, c, data[f"X_{c.upper()}"])
            _enc_anime(ph1, c, s, data[f"Y_{key.upper()}"])
            # reconstruct with audio / content
            ph1[f"y_{key}_rec_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_sty_{s}"])
            ph1[f"y_{key}_rec_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_sty_{s}"])
            # debug: zero content or zero style
            if self.tb_vis or self.generating:
                ph1[f"y_0{s}_vis_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_sty_{s}"], ["z_ctt"])
                ph1[f"y_{c}0_vis_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_sty_{s}"], ["z_sty"])
                ph1[f"y_{c}0_vis_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_sty_{s}"], ["z_sty"])
        # swap
        for key in ["iq", "jp"]:
            c, s = key[0], key[1]
            ph1[f"y_{key}_swp_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_sty_{s}"])
            ph1[f"y_{key}_swp_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_sty_{s}"])
        # set
        out["phase_1st"] = ph1

        # * Second Phase
        if not self.ablation.no_ph2:
            ph2 = dict()
            for key in ["iq", "jp"]:
                c, s = key[0], key[1]
                # copy audio latents from phase_1st
                ph2[f"z_ctt_aud_{c}"] = ph1[f"z_ctt_aud_{c}"]
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
                    ph2[f"y_0{s}_vis_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_sty_{s}"], ["z_ctt"])
                    ph2[f"y_{c}0_vis_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_sty_{s}"], ["z_sty"])
                    ph2[f"y_{c}0_vis_aud"] = _remix(ph2[f"z_ctt_aud_{c}"], ph2[f"z_sty_{s}"], ["z_sty"])
            # swap
            for key in ["ip", "jq"]:
                c, s = key[0], key[1]
                ph2[f"y_{key}_cyc_ani"] = _remix(ph2[f"z_ctt_ani_{c}"], ph2[f"z_sty_{s}"])
                ph2[f"y_{key}_cyc_aud"] = _remix(ph2[f"z_ctt_aud_{c}"], ph2[f"z_sty_{s}"])
            # set
            out["phase_2nd"] = ph2

        # copy vidx
        out["FACE_VIDX"] = FACE_VIDX
        out["NON_FACE_VIDX"] = NON_FACE_VIDX
        # fmt: on
        return out

    def reconstruct(self, idle_verts, **data):
        # fmt: off
        def _enc_audio(res, c, X):
            res[f"z_ctt_aud_{c}"] = self.animnet_decmp.encode_audio(X)

        def _enc_anime(res, c, s, Y):
            res[f"z_ctt_ani_{c}"], res[f"z_sty_{s}"] = self.animnet_decmp.decomp_offsets(Y)

        def _remix(z_ctt, z_sty, set_zeros=()):
            if 'z_ctt' in set_zeros: z_ctt = torch.zeros_like(z_ctt)
            if 'z_sty' in set_zeros: z_sty = torch.zeros_like(z_sty)
            return self.animnet_decmp.remix(z_ctt, z_sty, idle_verts)["out_verts"] - idle_verts.unsqueeze(1)

        out = dict()

        ph1 = dict()
        for key in ["ip", "jq"]:
            c, s = key[0], key[1]
            # encode
            _enc_audio(ph1, c, data[f"X_{c.upper()}"])
            _enc_anime(ph1, c, s, data[f"Y_{key.upper()}"])
            # reconstruct with audio / content
            ph1[f"y_{key}_rec_ani"] = _remix(ph1[f"z_ctt_ani_{c}"], ph1[f"z_sty_{s}"])
            ph1[f"y_{key}_rec_aud"] = _remix(ph1[f"z_ctt_aud_{c}"], ph1[f"z_sty_{s}"])
        # set
        out["phase_1st"] = ph1

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
        dat["idle_verts"] = idle_verts

        # output
        if self.ablation.e2e:
            out = dict()
            out["Y_IP"] = self.get_Y(dat["I"])
            out["Y_JQ"] = self.get_Y(dat["J"])
            out["X_I"] = dat["I"].get("audio_dict")
            out["X_J"] = dat["J"].get("audio_dict")
            out.update(self.reconstruct(idle_verts, **out))
        else:
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

    def on_validation_start(self):
        log.info("Validation start")
        self._valid_z_sty = None

    def on_validation_end(self):
        self._valid_z_sty = None

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

        if self.ablation.e2e:
            part = self.hparams.loss.mesh.part
            scale = loss_opts.rec * 0.5 * loss_opts.anime
            _loss_fn("ani:rec", out["phase_1st"]["y_ip_rec_ani"], out["Y_IP"], scale, part)
            _loss_fn("ani:rec", out["phase_1st"]["y_jq_rec_ani"], out["Y_JQ"], scale, part)
            scale = loss_opts.rec * 0.5 * loss_opts.audio
            _loss_fn("aud:rec", out["phase_1st"]["y_ip_rec_aud"], out["Y_IP"], scale, part)
            _loss_fn("aud:rec", out["phase_1st"]["y_jq_rec_aud"], out["Y_JQ"], scale, part)
            return sum_up_losses(ldict, wdict, training)

        # INFO: Reconstuction loss item, should reconstruct the target part (from config)
        if loss_opts.rec > 0:
            part = self.hparams.loss.mesh.part
            scale = loss_opts.rec * 0.5 * loss_opts.anime
            _loss_fn("ani:rec", out["phase_1st"]["y_ip_rec_ani"], out["Y_IP"], scale, part)
            _loss_fn("ani:rec", out["phase_1st"]["y_jq_rec_ani"], out["Y_JQ"], scale, part)
            scale = loss_opts.rec * 0.5 * loss_opts.audio
            _loss_fn("aud:rec", out["phase_1st"]["y_ip_rec_aud"], out["Y_IP"], scale, part)
            _loss_fn("aud:rec", out["phase_1st"]["y_jq_rec_aud"], out["Y_JQ"], scale, part)

        # INFO: The Loss for swap (1st phase), only supervise lower face part
        if loss_opts.swp > 0:
            # part = self.hparams.loss.mesh.part
            part = "face_below_eye"
            exkw = dict(scale_inv=0.0, scale_oth=0.0) if loss_opts.swp_only_low else dict()
            scale = loss_opts.swp * 0.5 * loss_opts.anime
            _loss_fn("ani:swp", out["phase_1st"]["y_iq_swp_ani"], out["Y_IQ"], scale, part, **exkw)
            _loss_fn("ani:swp", out["phase_1st"]["y_jp_swp_ani"], out["Y_JP"], scale, part, **exkw)
            scale = loss_opts.swp * 0.5 * loss_opts.audio
            _loss_fn("aud:swp", out["phase_1st"]["y_iq_swp_aud"], out["Y_IQ"], scale, part, **exkw)
            _loss_fn("aud:swp", out["phase_1st"]["y_jp_swp_aud"], out["Y_JP"], scale, part, **exkw)

        # INFO: The Loss for cyc (2nd phase), should reconstruct the target part (from config)
        if loss_opts.cyc > 0:
            part = self.hparams.loss.mesh.part
            scale = loss_opts.cyc * 0.5 * loss_opts.anime
            _loss_fn("ani:cyc", out["phase_2nd"]["y_ip_cyc_ani"], out["Y_IP"], scale, part)
            _loss_fn("ani:cyc", out["phase_2nd"]["y_jq_cyc_ani"], out["Y_JQ"], scale, part)
            scale = loss_opts.cyc * 0.5 * loss_opts.audio
            _loss_fn("aud:cyc", out["phase_2nd"]["y_ip_cyc_aud"], out["Y_IP"], scale, part)
            _loss_fn("aud:cyc", out["phase_2nd"]["y_jq_cyc_aud"], out["Y_JQ"], scale, part)
        # fmt: on

        if self.ablation.no_reg:
            return sum_up_losses(ldict, wdict, training)

        # INFO: Regularization of latents
        def _l2_reg(a, b):
            return ((a - b) ** 2).mean() * 0.5

        if loss_opts.reg_z_ctt_close > 0 and (not self.ablation.no_ph2):
            item_ctt_i = _l2_reg(out["phase_1st"]["z_ctt_ani_i"], out["phase_2nd"]["z_ctt_ani_i"])
            item_ctt_j = _l2_reg(out["phase_1st"]["z_ctt_ani_j"], out["phase_2nd"]["z_ctt_ani_j"])
            ldict["reg:z_ctt_close"] = item_ctt_i + item_ctt_j
            wdict["reg:z_ctt_close"] = loss_opts.reg_z_ctt_close

        if loss_opts.reg_z_sty_close > 0 and (not self.ablation.no_ph2):
            item_sty_p = _l2_reg(out["phase_1st"]["z_sty_p"], out["phase_2nd"]["z_sty_p"])
            item_sty_q = _l2_reg(out["phase_1st"]["z_sty_q"], out["phase_2nd"]["z_sty_q"])
            ldict["reg:z_sty_close"] = item_sty_p + item_sty_q
            wdict["reg:z_sty_close"] = loss_opts.reg_z_sty_close

        if loss_opts.reg_z_aud_close > 0:  # and self.using_audio:
            item_ph1 = _l2_reg(out["phase_1st"]["z_ctt_ani_i"], out["phase_1st"]["z_ctt_aud_i"]) + _l2_reg(
                out["phase_1st"]["z_ctt_ani_j"], out["phase_1st"]["z_ctt_aud_j"]
            )
            item_ph2 = (
                (
                    _l2_reg(out["phase_2nd"]["z_ctt_ani_i"], out["phase_2nd"]["z_ctt_aud_i"])
                    + _l2_reg(out["phase_2nd"]["z_ctt_ani_j"], out["phase_2nd"]["z_ctt_aud_j"])
                )
                if (not self.ablation.no_ph2)
                else 0.0
            )
            ldict["reg:z_aud_close"] = item_ph1 + item_ph2
            wdict["reg:z_aud_close"] = loss_opts.reg_z_aud_close

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
        if datamodule is not None:
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
                    generator.generate_for_dataset(
                        dataset=validset, set_type="valid", seq_list=seq_list, **shared_kwargs
                    )

                if self.hparams.visualizer.generate_train and trainset is not None:
                    seq_list = self.hparams.visualizer.get("trainset_seq_list") or trainset_seq_list
                    generator.generate_for_dataset(
                        dataset=trainset, set_type="train", seq_list=seq_list, **shared_kwargs
                    )

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
        # if constants.MODE == "train":
        #     from .do_tsne import do_tsne

        #     do_tsne(epoch, self.hparams.visualizer.speaker)


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                               Pre-compute Paddings                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #


""" Run by '_init_' in config.model.  It will compute all padding information for sequential modules.  """


def pre_compute_paddings(config: DictConfig):
    config = config.animnet

    if config.sequential.using == "conv":
        from ..modules.seq_conv import SeqConv

        SeqConv.compute_padding_information(config)
    elif config.sequential.using == "xfmr":
        from src.engine.misc.table import Table

        with open_dict(config):
            config.src_seq_pads = tuple([config.sequential.xfmr.win_size - 1, 0])

        table = Table(alignment=("left", "middle"))
        table.add_row("animnet.src_seq_frames", config.src_seq_frames)
        table.add_row("animnet.tgt_seq_frames", config.tgt_seq_frames)
        table.add_row("animnet.src_seq_pads", str(config.src_seq_pads))
        log.info("Pre-Computed Paddings:\n{}".format(table))
    else:
        raise NotImplementedError()
