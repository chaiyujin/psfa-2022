import cv2
import numpy as np
import torch

from src.data.viseme import id_to_phoneme, id_to_viseme
from src.engine.mesh_renderer import render, render_heatmap
from src.engine.painter import Text, color_mapping, draw_canvas, heatmap, put_texts
from src.metrics.image import MaskedPSNR_255
from src.projects.anim import register_painter


@register_painter
def demo_columns(cls, hparams, batch, results, bi, fi, A=256):
    def _metrics_txt(tag, fake, real, mask, color=cls.colors.white):
        if fake is None or real is None:
            return None
        psnr = (
            MaskedPSNR_255.compute(
                torch.from_numpy(fake.transpose(2, 0, 1) * 255),
                torch.from_numpy(real.transpose(2, 0, 1) * 255),
                torch.from_numpy(mask.transpose(2, 0, 1)),
            )
            .mean()
            .item()
        )
        return (tag, color, f"{psnr:.1f}", color)

    # fmt: off
    def _b(key): return cls.get(batch, key, bi, fi)  # noqa
    def _h(key): return cls.get(results, key, bi, fi)  # noqa

    v_frame   = _b("image")
    face_mask = _b("face_mask")
    if v_frame is not None:
        v_frame = v_frame.transpose(1, 2, 0)
        face_mask = face_mask.transpose(1, 2, 0).repeat(3, -1)

    # get verts
    dfrm_verts = _h("animnet.dfrm_verts")
    trck_verts = _h("animnet.trck_verts")
    REAL_verts = _h("animnet.REAL_verts")
    # get neural renderer
    nr_dfrm_verts = _h("animnet.imspc.nr_fake")
    nr_trck_verts = _h("animnet.imspc-trck.nr_fake")
    # fmt: on

    # * prepare
    titles, rows, txts = [], [], []

    # * Columns check
    vfrm_column = v_frame is not None and (nr_dfrm_verts is not None or nr_trck_verts is not None)
    dfrm_column = dfrm_verts is not None
    trck_column = trck_verts is not None and hparams.visualizer.get("demo_columns", dict()).get("trck", True)
    REAL_column = REAL_verts is not None

    # fmt: off
    if vfrm_column: titles.append("Video (Masked)")
    if dfrm_column: titles.append("Deformed")
    if trck_column: titles.append("Tracked")
    if REAL_column: titles.append("REAL")
    # fmt: on

    v_frame = cls.normalize_img(v_frame)
    nr_dfrm_verts = cls.normalize_img(nr_dfrm_verts)
    nr_trck_verts = cls.normalize_img(nr_trck_verts)

    # * Row 0
    if v_frame is not None and (nr_dfrm_verts is not None or nr_trck_verts is not None):
        row, txt = [], []

        if vfrm_column:
            mask = face_mask
            if _h("animnet.imspc.mask_weye") is not None:
                mask = np.logical_and(_h("animnet.imspc.mask_weye"), face_mask)
            row.append(cls.masked(v_frame, mask))
            txt.append(None)

        if dfrm_column:
            if nr_dfrm_verts is not None:
                nr_fake_img = cls.masked(nr_dfrm_verts, _h("animnet.imspc.mask_weye"))
                nr_fake_txt = _metrics_txt("Neural Render", nr_dfrm_verts, v_frame, _h("animnet.imspc.mask_neye"))
                row.append(nr_fake_img)
                txt.append(nr_fake_txt)
            else:
                row.append(None)
                txt.append(None)

        if trck_column:
            if nr_trck_verts is not None:
                nr_fake_img = cls.masked(nr_trck_verts, _h("animnet.imspc-trck.mask_weye"))
                nr_fake_txt = _metrics_txt("Neural Render", nr_trck_verts, v_frame, _h("animnet.imspc-trck.mask_neye"))
                row.append(nr_fake_img)
                txt.append(nr_fake_txt)
            else:
                row.append(None)
                txt.append(None)

        if REAL_column:
            row.append(None)
            txt.append(None)

        rows.append(row)
        txts.append(txt)

    # * Row 1 and Row 2
    # for side in [False, "45", True]:
    # for side in [False, True]:
    for side in [False]:
        row, txt = [], []

        if vfrm_column:
            if not side:
                if hparams.loss.rendering is not None:
                    # * Row1, landmarks for v_frame
                    fake_pts = _h("animnet.imspc.lmks_fw75")
                    real_pts = _b("lmks_fw75")
                    im_lmks = v_frame.copy()
                    im_lmks = cls.draw_landmarks(im_lmks, fake_pts, real_pts, hparams.loss.rendering.lmks_weights)
                    row.append(im_lmks)
                    txt.append(
                        ("Landmarks", cls.colors.white, "Real: Green", cls.colors.green, "Pred: Red", cls.colors.red)
                    )
                else:
                    row.append(None)
                    txt.append(None)
            else:
                audio_feat = _b("audio_dict.deepspeech")[..., 0]
                audio_feat = np.flip(audio_feat, axis=1)
                audio_feat = np.transpose(audio_feat, (1, 0))
                im = color_mapping(audio_feat)
                feat_name = "DeepSpeech"

                attn = _h("animnet.align_dict.audio_attn")
                if attn is not None:
                    attn = np.repeat(attn, 5, 0)
                    im1 = color_mapping(attn, vmin=0, vmax=1)
                    im = np.concatenate((im, im1), axis=0)
                    feat_name += " - Attn"

                row.append(im)
                txt.append((feat_name, cls.colors.white))

        if dfrm_column:
            im = render(dfrm_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if trck_column:
            im = render(trck_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if REAL_column:
            im = render(REAL_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        rows.append(row)
        txts.append(txt)

    if _h("animnet.coarse_attn") is not None or _h("animnet.fine_attn") is not None:
        row, txt = [], []

        if vfrm_column:
            row.append(None)
            txt.append(None)

        if dfrm_column:
            fine_attn = _h("animnet.fine_attn")
            if fine_attn is not None:
                im = heatmap(fine_attn[:, None], vmin=0, vmax=1)  # type: ignore
                im = cv2.resize(im, (A, A), interpolation=cv2.INTER_NEAREST)
                row.append(im)
                txt.append(("Fine Attn", cls.colors.white))
            else:
                row.append(None)
                txt.append(None)

        if trck_column:
            row.append(None)
            txt.append(None)

        if REAL_column:
            row.append(None)
            txt.append(None)

        rows.append(row)
        txts.append(txt)

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_mixer(cls, hparams, batch, results, bi, fi, A=256):

    # fmt: off
    def _b(key): return cls.get(batch, key, bi, fi)  # noqa
    def _h(key): return cls.get(results, key, bi, fi)  # noqa

    # get verts
    REAL_verts = _h("animnet.REAL_verts")
    dfrm_verts = _h("animnet.dfrm_verts")
    dfrf_verts = _h("animnet.dfrm_verts_refer")
    refr_verts = _h("animnet.REFR_verts")
    adns_verts = _h("animnet.dfrm_verts_audio_nostyle")
    rfns_verts = _h("animnet.dfrm_verts_refer_nostyle")

    # * prepare
    titles, rows, txts = [], [], []

    # * Columns check
    REAL_col = REAL_verts is not None
    dfrm_col = dfrm_verts is not None
    dfrf_col = dfrf_verts is not None
    refr_col = refr_verts is not None
    adns_col = adns_verts is not None
    rfns_col = rfns_verts is not None

    if REAL_col: titles.append(("REAL", cls.colors.green))
    else       : titles.append(("", cls.colors.white))
    if dfrm_col: titles.append(("Output (Final)", cls.colors.blue))
    if dfrf_col: titles.append(("Output (ctt_refer)", cls.colors.white))
    if refr_col: titles.append(("Refer", cls.colors.white))
    if adns_col: titles.append(("ctt_audio nostyle", cls.colors.white))
    if rfns_col: titles.append(("ctt_refer nostyle", cls.colors.white))
    # fmt: on

    # * Row 1 and Row 2
    for side in [False, True]:
        row, txt = [], []

        if REAL_col:
            im = render(REAL_verts, A=A, side=side)
            row.append(im)
            txt.append(None)
        else:
            row.append(None)
            txt.append(None)

        if dfrm_col:
            im = render(dfrm_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if dfrf_col:
            im = render(dfrf_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if refr_col:
            im = render(refr_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if adns_col:
            im = render(adns_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        if rfns_col:
            im = render(rfns_verts, A=A, side=side)
            row.append(im)
            txt.append(None)

        rows.append(row)
        txts.append(txt)

    # * attention or gates
    if _h("animnet.vis_label_prob_audio") is not None or _h("animnet.vis_label_prob_refer") is not None:
        using_label = hparams.visualizer.demo_mixer.label
        assert using_label in ["viseme", "phoneme"]
        fn = id_to_viseme if using_label == "viseme" else id_to_phoneme
        ids_real = _h("animnet.vis_label_ids_real")

        def _append_prb_ids(name, prb, ids_pred):
            if prb is None:
                return None

            im = heatmap(prb[:, None], vmin=0, vmax=1)  # type: ignore
            im = cv2.resize(im, (A, A), interpolation=cv2.INTER_NEAREST)
            this_txt_list = [Text(name, (A // 2, 0), ("center", "top"), cls.colors.white)]
            if ids_pred is not None:
                this_txt_list.append(Text(f"P: {fn(ids_pred)}", (A // 2, 13), ("center", "top"), cls.colors.red))
            if ids_real is not None:
                this_txt_list.append(Text(f"R: {fn(ids_real)}", (A // 2, 26), ("center", "top"), cls.colors.green))
            im = put_texts(im, this_txt_list, font_size=12)
            return im

        row, txt = [], []

        if REAL_col:
            row.append(None)
            txt.append(None)
        else:
            row.append(None)
            txt.append(None)

        if dfrm_col:
            prb_audio = _h("animnet.vis_label_prob_audio")
            ids_audio = _h("animnet.vis_label_ids_audio")
            row.append(_append_prb_ids("Audio", prb_audio, ids_audio))
            txt.append(None)

        if dfrf_col:
            prb_refer = _h("animnet.vis_label_prob_refer")
            ids_refer = _h("animnet.vis_label_ids_refer")
            row.append(_append_prb_ids("RefXY", prb_refer, ids_refer))
            txt.append(None)

        if refr_col:
            row.append(None)
            txt.append(None)

        if adns_col:
            row.append(None)
            txt.append(None)

        if rfns_col:
            row.append(None)
            txt.append(None)

        rows.append(row)
        txts.append(txt)

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas
