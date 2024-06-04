import json
import os

import cv2
import numpy as np
import torch
from torch import Tensor

from assets import ASSETS_ROOT, get_vocaset_template_triangles


def get_flame_landmark_barycoords():
    flm_tris = get_vocaset_template_triangles()
    with open(os.path.join(ASSETS_ROOT, "flame_lmk_ibug68.json")) as fp:
        data = json.load(fp)
        lmk_vidx = np.asarray([[flm_tris[ti][i] for i in range(3)] for ti, _, _ in data], dtype=np.int64)  # type: ignore
        lmk_wght = np.asarray([[w0, w1, 1.0 - w0 - w1] for _, w0, w1 in data], dtype=np.float32)  # type: ignore
    lmk_barycoords = (
        torch.tensor(lmk_vidx, device="cpu"),
        torch.tensor(lmk_wght, device="cpu"),
    )
    return lmk_barycoords


def get_flame_jaw_open():
    delta = np.load(os.path.join(ASSETS_ROOT, "jaw_open_offsets.npy"))
    return torch.tensor(delta, dtype=torch.float32, device="cpu")


def things_for_morphing_inner_mouth(innm_dir):
    A = np.load(os.path.join(innm_dir, "A.npy"))
    B = np.load(os.path.join(innm_dir, "B.npy"))
    A = torch.tensor(A, dtype=torch.float32, device="cpu")
    B = torch.tensor(B, dtype=torch.float32, device="cpu")
    lmk_vidx, lmk_wght = get_flame_landmark_barycoords()
    return A, B, lmk_vidx, lmk_wght


def get_inner_mouth_offsets(verts, idle_verts, A, B, lmk_vidx, lmk_wght, speaker) -> Tensor:
    def _get_flame_lmks(verts):
        assert verts.ndim == 3  # N,V,3
        N, _, _ = verts.shape
        P, _ = lmk_vidx.shape
        v_faces = verts[:, lmk_vidx.view(-1), :]  # N,P*3,3
        v_faces = v_faces.view(N, P, 3, 3)  # N,P,3,3
        v_faces = v_faces * lmk_wght[None, :, :, None]
        v = v_faces.sum(dim=-2)
        return v

    # get jaw lmk movement
    jaw_move = _get_flame_lmks(verts - idle_verts)
    jaw_move = jaw_move[:, 8, 1:]  # don't consider x-axis

    # solve coeff
    y = jaw_move.t()
    if torch.__version__ >= "1.9":
        x = torch.linalg.solve(A, y).t()  # type: ignore
    else:
        x = torch.solve(y, A).solution.t()  # type: ignore

    if speaker == "m001_trump":
        x = x * torch.clamp(x, min=0.5, max=0.8)
    elif speaker == "m000_obama":
        x = x * x * x * 0.5
    elif speaker == "f000_watson":
        x = x * torch.clamp(x, min=0.5, max=0.8)
    elif speaker == "f001_clinton":
        x = x * torch.clamp(x, min=0.5, max=0.8) * 0.9
    elif speaker == "f002_holsman":
        x = x * torch.clamp(x, min=0.0, max=1.0) * 1.1
    elif speaker == "m002_taruvinga":
        x = x * torch.clamp(x, min=0.2, max=0.8) * 0.8
    elif speaker == "m003_iphoneXm0":
        x = x * x * 0.4
    else:
        raise NotImplementedError("Unknown speaker for get_inner_mouth_offsets: {}".format(speaker))

    # get delta
    delta_innm = torch.matmul(x, B).view(x.shape[0], -1, 3)  # type: ignore
    return delta_innm


def fill_hole(trans):
    im_th = trans[..., 0]
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)  # type: ignore
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # type: ignore
    im_out = im_th | im_floodfill_inv
    return im_out[:, :, None]
