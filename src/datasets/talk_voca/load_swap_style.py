import os

import numpy as np
import torch

from src.data.mesh import load_mesh
from src.datasets.utils_load import load_vertices
from src.datasets.utils_seq import Range, avoffset_to_frame_delta
from src.engine.misc import filesys


def load_offsets_swap(config, swap_src_path, swap_info, frm_range: Range, all_frames: bool = False):
    assert isinstance(swap_src_path, str)
    assert os.path.exists(swap_src_path)

    # Get the total frames of swapping source.
    # The frames is under swap_info.data_fps, maybe different from frm_range.fps
    if os.path.isdir(swap_src_path):
        n_swp_frames = len(filesys.find_files(swap_src_path, r".*\.npy", False, False))
    elif os.path.splitext(swap_src_path)[1] == ".npy":
        n_swp_frames = len(np.load(swap_src_path, mmap_mode="r"))
    else:
        raise NotImplementedError("Unknown swapping source: '{}'".format(swap_src_path))

    # Get swap range
    swp_range = frm_range.copy()
    if not all_frames:
        # just some frames to match with frm_range, for on-fly validation in training phase
        while swp_range.stt_idx + swp_range.n_frames > n_swp_frames:
            swp_range.stt_idx = max(0, swp_range.stt_idx - n_swp_frames)
    else:
        # load the entire sequence, for inferring style vector during video generation.
        swp_range.stt_idx = 0
        swp_range.n_frames = n_swp_frames
    # print("Validata swap data at {} fps, wanna {} fps".format(swap_info.data_fps, frm_range.fps))

    # Load the offsets (maybe meshes)
    offsets_swap = load_vertices(
        config,
        swp_range,
        a=0,
        data_fps=swap_info["data_fps"],  # NOTE: the given data_fps
        max_frames=n_swp_frames,  # the maximum number of frames of source (under data_fps)
        data_dir=os.path.dirname(swap_src_path),
        sub_path=os.path.basename(swap_src_path),
        frame_delta=avoffset_to_frame_delta(swap_info.get("avoffset", 0)),
    )

    # Handle the case, where given source is a directory of meshes vertices.
    # We remove the identity template to get offsets.
    if os.path.basename(swap_src_path) == "meshes":
        assert "idle" in swap_info
        spk_idle, _, _ = load_mesh(swap_info["idle"])
        offsets_swap -= torch.tensor(spk_idle[None, ...], dtype=torch.float)
    return offsets_swap


def load_and_concat_all_offsets_swap(config, sources, swap_info, frm_range: Range, verbose: bool = True):
    if isinstance(sources, str):
        sources = [sources]
    res_list = []
    for src in sources:
        if not os.path.exists(src):
            if verbose:
                print(f"Swap source '{src}' doesn't exist, skip.")
            continue
        off = load_offsets_swap(config, src, swap_info, frm_range, all_frames=True)
        res_list.append(off)
    return torch.cat(res_list)
