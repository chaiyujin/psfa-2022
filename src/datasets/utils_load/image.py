import os
from typing import Dict

import torch
import torchvision as tv
from torch import Tensor

from src.data import image as imutils
from src.datasets.utils_seq import Range, parse_float_index


def load_images(
    config, frm_range: Range, a: float, data_dir: str, data_fps: float, max_frames: int, frame_delta: int = 0, **kwargs
) -> Dict[str, Tensor]:

    im_size = config.video.image_size
    transforms = [tv.transforms.Resize(im_size)]

    def _frm_to_data_idx(frm_idx):
        return frm_idx * data_fps / frm_range.fps

    def _load_frame(index):
        img_path = os.path.join(data_dir, f"images/{index:d}.png")
        msk_path = os.path.join(data_dir, f"image_masks/{index:d}_face.png")
        inm_path = os.path.join(data_dir, f"image_masks/{index:d}_innm.png")
        lip_path = os.path.join(data_dir, f"image_masks/{index:d}_lips.png")
        eye_path = os.path.join(data_dir, f"image_masks/{index:d}_eyes.png")
        assert os.path.exists(img_path)
        img = imutils.load_th(img_path, transforms, vrange=(-1, 1))  # -1 ~ 1
        msk = imutils.load_th(msk_path, transforms) if os.path.exists(msk_path) else torch.ones_like(img)
        # mouth
        inm = imutils.load_th(inm_path, transforms) if os.path.exists(inm_path) else torch.ones_like(img)
        lip = imutils.load_th(lip_path, transforms) if os.path.exists(lip_path) else torch.ones_like(img)
        mth = torch.clamp(inm + lip, min=0, max=1)

        # eye = imutils.load_th(eye_path, transforms) if os.path.exists(eye_path) else torch.ones_like(img)

        return img[:3], msk[:1], mth[:1]

    # start to read
    list_img, list_msk, list_mth = [], [], []
    prv_idx, prv_img, prv_msk, prv_mth = None, None, None, None
    for i in frm_range:
        # * interpolate 'a'
        i = i + a
        # get index under data_fps
        index = _frm_to_data_idx(i)
        # shift frame index
        index += frame_delta
        # convert float index
        ifrm, jfrm, alpha = parse_float_index(index, max_frames=max_frames)
        # ! Since we don't interpolate image,
        # ! we simply use the one close to the required.
        idx = ifrm if alpha <= 0.5 else jfrm
        # load and cache
        if prv_idx is not None and idx == prv_idx:
            img, msk, mth = (prv_img, prv_msk, prv_mth)
        else:
            img, msk, mth = _load_frame(idx)
        prv_idx, prv_img, prv_msk, prv_mth = idx, img, msk, mth
        # append
        list_img.append(img)
        list_msk.append(msk)
        list_mth.append(mth)

    # finish, stack as tensor
    ret = dict()
    ret["image"] = torch.stack(list_img, dim=0)
    ret["face_mask"] = torch.stack(list_msk, dim=0)
    ret["mouth_mask"] = torch.stack(list_mth, dim=0)

    # post-process mask
    for key in ret:
        if key.find("mask") >= 0:
            ret[key] = ret[key] >= 0.05

    return ret
