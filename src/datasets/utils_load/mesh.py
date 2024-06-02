import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from src.datasets.utils_seq import Range, interp, interp_seq, parse_float_index


# sub_path can be a filename or a dirname.
def load_vertices(
    config,
    frm_range: Range,
    a: float,
    data_dir: str,
    data_fps: float,
    max_frames: int,
    sub_path: str,
    frame_delta: float = 0,
    npz_key: Optional[str] = None,
    dtw_idx_npy: Optional[str] = None,
    **kwargs,
) -> Tensor:
    def _frm_to_data_idx(frm_idx):
        return frm_idx * data_fps / frm_range.fps

    def _npload(i_frame):
        if i_frame < 0:
            return np.zeros_like(np.load(os.path.join(data_dir, sub_path, f"{0:06d}.npy")))
        if i_frame >= max_frames:
            i_frame = max_frames - 1
        return np.load(os.path.join(data_dir, sub_path, f"{i_frame:06d}.npy"))

    isdir = os.path.isdir(os.path.join(data_dir, sub_path))
    isnpy = os.path.isfile(os.path.join(data_dir, sub_path)) and os.path.splitext(sub_path)[1] == ".npy"
    isnpz = os.path.isfile(os.path.join(data_dir, sub_path)) and os.path.splitext(sub_path)[1] == ".npz"
    dirpath = os.path.join(data_dir, sub_path)
    assert isdir or isnpy or isnpz, "Given 'sub_path' ({}) is not .npy, .npz or dirname!".format(dirpath)
    if isnpy:
        seq = np.load(os.path.join(data_dir, sub_path), mmap_mode="r")
        max_frames = min(max_frames, len(seq))
    elif isnpz:
        npz_data = np.load(os.path.join(data_dir, sub_path), mmap_mode="r")
        if npz_key is None:
            assert len(npz_data) == 1, "Given 'sub_path' ({}) is .npz, but 'npz_key' is not given!".format(sub_path)
            npz_key = list(npz_data.keys())[0]
        seq = npz_data[npz_key]
        max_frames = min(max_frames, len(seq))
    else:
        seq = None

    dtw_idx = None
    if dtw_idx_npy is not None:
        dtw_idx = np.load(dtw_idx_npy)

    data_list = []
    for i in frm_range:
        # * interpolate 'a'
        i = i + a
        # get index under data_fps
        index = _frm_to_data_idx(i)
        # frame delta
        index += frame_delta

        # load data and interploate
        if dtw_idx is None:
            # interpolate index
            ifrm, jfrm, alpha = parse_float_index(index, max_frames=max_frames)
            # normally load
            if isdir:
                data_list.append(interp(_npload(ifrm), _npload(jfrm), alpha))
            else:
                assert isnpy or isnpz
                data_list.append(interp_seq(seq, ifrm, jfrm, alpha))
        else:

            def _frm_idx_to_mel_idx(ifrm):
                sec = ifrm / data_fps
                return sec / kwargs["mel_hop"]

            def _mel_idx_to_frm_idx(imel):
                # get the sec for this reader
                sec = imel * kwargs["mel_hop"]
                ifrm = int(np.round(sec * data_fps))
                return np.clip(ifrm, 0, max_frames - 1)

            # interpolate index
            mel_idx = _frm_idx_to_mel_idx(index)
            imel, jmel, alpha = parse_float_index(mel_idx, max_frames=len(dtw_idx))
            # get index from dtw_idx
            # fmt: off
            data_i, cnt_i = 0, 0
            data_j, cnt_j = 0, 0
            for j in range(*dtw_idx[imel]):
                k = _mel_idx_to_frm_idx(j)
                data_i += _npload(k) if isdir else seq[k]
                cnt_i += 1
            for j in range(*dtw_idx[jmel]):
                k = _mel_idx_to_frm_idx(j)
                data_j += _npload(k) if isdir else seq[k]
                cnt_j += 1
            data_i /= cnt_i
            data_j /= cnt_j
            # fmt: on
            data_list.append(interp(data_i, data_j, alpha))

    return torch.tensor(np.asarray(data_list, dtype=np.float32))
