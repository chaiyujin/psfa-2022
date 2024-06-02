import os
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from src.datasets.utils_seq import Range, interp, interp_seq


def load_tracked_data(
    config,
    frm_range: Range,
    a: float,
    data_dir: str,
    sub_path: str,
    data_fps: float,
    max_frames: int,
    frame_delta: int = 0,
    **kwargs,
) -> Dict[str, Tensor]:
    def _frm_to_data_idx(frm_idx):
        return frm_idx * data_fps / frm_range.fps

    seq_dir = os.path.join(data_dir, sub_path)
    assert os.path.isdir(seq_dir)

    # iter frame range
    list_cam, list_rot, list_tsl = [], [], []
    for _i in frm_range:
        # * interploate 'a'
        _i = _i + a
        # get index under data_fps
        index = _frm_to_data_idx(_i)
        # shift index
        index += frame_delta
        # interpolate
        ifrm, jfrm, alpha = int(index), int(index) + 1, index - int(index)
        idx = np.clip(ifrm, 0, max_frames - 1)
        jdx = np.clip(jfrm, 0, max_frames - 1)
        # npz_data
        data0 = np.load(os.path.join(seq_dir, f"{idx:06d}.npz"))
        data1 = np.load(os.path.join(seq_dir, f"{jdx:06d}.npz"))
        list_cam.append(interp(data0["cam"], data1["cam"], alpha))
        list_rot.append(interp(data0["rot"], data1["rot"], alpha))
        list_tsl.append(interp(data0["tsl"], data1["tsl"], alpha))

    # fmt: off
    ret = dict()
    ret["cam"] = torch.tensor(np.asarray(list_cam, dtype=np.float32))
    ret["rotat"] = torch.tensor(np.asarray(list_rot, dtype=np.float32))
    ret["transl"] = torch.tensor(np.asarray(list_tsl, dtype=np.float32))
    # fmt: on
    return ret


def load_landmarks(
    config,
    frm_range: Range,
    a: float,
    data_dir: str,
    sub_path: str,
    data_fps: float,
    max_frames: int,
    frame_delta: int = 0,
    **kwargs,
) -> Dict[str, Tensor]:
    def _frm_to_data_idx(frm_idx):
        return frm_idx * data_fps / frm_range.fps

    seq_marks = np.load(os.path.join(data_dir, sub_path))

    # iter frame range
    list_marks = []
    for _i in frm_range:
        # * interploate 'a'
        _i = _i + a
        # get index under data_fps
        index = _frm_to_data_idx(_i)
        # shift index
        index += frame_delta
        # interpolate
        ifrm, jfrm, alpha = int(index), int(index) + 1, index - int(index)
        idx = np.clip(ifrm, 0, max_frames - 1)
        jdx = np.clip(jfrm, 0, max_frames - 1)
        list_marks.append(interp_seq(seq_marks, idx, jdx, alpha))

    return torch.tensor(np.asarray(list_marks, dtype=np.float32))
