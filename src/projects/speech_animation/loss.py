from functools import lru_cache

import cv2
import torch
import torch.nn.functional as F

import assets
from assets import EYES_VIDX
from src.engine.mesh_renderer import render
from src.engine.seq_ops import fold


@lru_cache(maxsize=5)
def _get_vidx(part):
    sel_vidx = getattr(assets, part.upper() + "_VIDX")
    inv_vidx = [x for x in range(3931) if (x not in sel_vidx) and (x not in EYES_VIDX)]
    oth_vidx = [x for x in range(3931) if (x not in sel_vidx) and (x not in inv_vidx)]

    all_vidx = sel_vidx + inv_vidx + oth_vidx
    assert sorted(all_vidx) == list(range(3931))

    # vert = get_vocaset_template_vertices().astype('float32')
    # vert1 = vert.copy(); vert1[face_vidx] = 0
    # vert2 = vert.copy(); vert2[non_face_vidx] = 0
    # vert3 = vert.copy(); vert3[other_vidx] = 0
    # im0 = render(vert)
    # im1 = render(vert1)
    # im2 = render(vert2)
    # im3 = render(vert3)
    # cv2.imshow('img0', im0)
    # cv2.imshow('img1', im1)
    # cv2.imshow('img2', im2)
    # cv2.imshow('img3', im3)
    # cv2.waitKey()
    # quit(1)

    return sel_vidx, inv_vidx, oth_vidx, len(sel_vidx) + len(inv_vidx) + len(oth_vidx)


def compute_mesh_loss(
    loss_opts,
    inp_verts,
    tar_verts,
    part: str,
    scale_inv=None,
    scale_oth=None,
    scale_pos=None,
    scale_mot=None,
    scale_lip=None,
    want_mean_pos=False,
    want_zero_mot=False,
    key_fmt="mesh:{}",
):
    # Guard
    if loss_opts is None:
        return {}, {}

    # Init the loss and weights dict
    ldict, wdict = dict(), dict()

    # INFO: Get vidx of selected, inverse (no eye), other.
    sel_vidx, inv_vidx, oth_vidx, cnt_all_vidx = _get_vidx(part)
    assert cnt_all_vidx == 3931

    # fmt: off
    if scale_pos is None: scale_pos = loss_opts.scale_pos
    if scale_mot is None: scale_mot = loss_opts.scale_motion
    if scale_lip is None: scale_lip = loss_opts.scale_lip_diff
    if scale_inv is None: scale_inv = loss_opts.scale_inv_part
    if scale_oth is None: scale_oth = loss_opts.scale_remained
    # fmt: on

    def _lip_diff(inp):
        inp = inp * 1000  # m -> mm
        return inp[..., assets.INN_LIP_UPPER_VIDX, :] - inp[..., assets.INN_LIP_LOWER_VIDX, :]

    def _loss_fn(inp, tar):
        # m -> mm
        inp = inp * 1000
        tar = tar * 1000

        def _item_part(vidx):
            if len(vidx) > 0:
                return F.mse_loss(inp[..., vidx, :], tar[..., vidx, :], reduction="none").sum(-1).sum(-1)
            else:
                return 0.0

        # selected part
        item0 = _item_part(sel_vidx)
        # inversed part (but without eyes)
        item1 = _item_part(inv_vidx) * scale_inv
        # other vertices (without eyeballs)
        item2 = _item_part(oth_vidx) * scale_oth
        # sumup, and average
        item = (item0 + item1 + item2) / float(cnt_all_vidx)
        return item

    # > reg track smooth, only for sequential data
    if scale_mot > 0 and inp_verts.ndim == 4:  # N,L,V,3
        new_motion = inp_verts[:, 1:] - inp_verts[:, :-1]  # x1 - x0
        ref_motion = tar_verts[:, 1:] - tar_verts[:, :-1]  # x1 - x0
        if want_zero_mot:
            ref_motion = torch.zeros_like(ref_motion)
        item_m = _loss_fn(new_motion, ref_motion)
        ldict[key_fmt.format("motion")] = item_m
        wdict[key_fmt.format("motion")] = scale_mot

    # > reg track
    if scale_pos > 0:
        ref_verts = tar_verts
        if want_mean_pos:
            assert ref_verts.ndim == 4
            ref_verts = torch.mean(ref_verts, dim=1, keepdim=True).expand_as(tar_verts)
        item_p = _loss_fn(inp_verts, ref_verts)
        ldict[key_fmt.format("pos")] = item_p
        wdict[key_fmt.format("pos")] = scale_pos

    # lip diff: only used when pos is enabled
    if scale_lip:
        item_lip = F.mse_loss(_lip_diff(inp_verts), _lip_diff(tar_verts), reduction="none")
        ldict[key_fmt.format("lip_diff")] = item_lip.sum(-1).mean(-1)
        wdict[key_fmt.format("lip_diff")] = scale_lip

    return ldict, wdict


def compute_label_ce(loss_opts, pred_logits, real_ids, using_label, tag):
    ldict, wdict = dict(), dict()

    if loss_opts is None or pred_logits is None or real_ids is None:
        return ldict, wdict

    # * probability of visemes
    # if data.get("viseme_ids") is not None
    assert pred_logits.ndim == 3 and real_ids.ndim == 2
    assert pred_logits.shape[1] == real_ids.shape[1]
    pred_logits, _ = fold(pred_logits)
    real_ids, _ = fold(real_ids)

    ldict[f"lbl-ce:{using_label}-{tag}"] = F.cross_entropy(pred_logits, real_ids, reduction="none")
    wdict[f"lbl-ce:{using_label}-{tag}"] = loss_opts.get(f"{using_label}_ce")

    return ldict, wdict
