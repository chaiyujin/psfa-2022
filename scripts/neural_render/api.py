import os
from glob import glob
from shutil import copyfile, rmtree
from typing import Any, Optional

import cv2
import numpy as np
import torch
from tqdm import trange

from .utils import T_Tensor, fetch_batch, fetch_list, load_data, load_renderer, to_device, to_tensor


def to_uint8_image(x, flip):
    assert x.ndim == 3
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if x.shape[0] in [1, 2, 3, 4]:
        x = x.transpose(1, 2, 0)  # HWC
    x = x * 0.5 + 0.5  # 0 ~ 1
    x = np.clip(x * 255, 0, 255).astype(np.uint8)  # 0 ~ 255
    if flip:
        x = x[..., [2, 1, 0]]
    return x


def lossless_image(save_path, img):
    cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def lossless_video(img_fmt, out_vpath, apath=None):
    """Generate lossless video, use vlc to playback"""

    cmd = f"ffmpeg -y -loglevel error -thread_queue_size 8192 -i '{img_fmt}'"
    if apath is not None:
        cmd += f" -i '{apath}'"
    # crf = 0 is lossless for 8-bit video
    cmd += f" -vf fps=25 -c:v libx264rgb -crf 0 -pix_fmt rgb24 -shortest '{out_vpath}'"
    assert os.system(cmd) == 0


###############################################################################
# Entrance function
###############################################################################


def neural_render(name: str, **kwargs):
    if name == "ours":
        return render_ours(**kwargs)
    else:
        raise NotImplementedError("Unknown nr: {}".format(name))


###############################################################################
# Our Neural Renderer
###############################################################################


@torch.no_grad()
def render_ours(
    out_vpath: str,
    verts: T_Tensor,
    idle_verts: T_Tensor,
    reenact_video: str,
    reenact_data_dir: str,
    model_path: str,
    cache_reenact: bool = True,
    cache_model: bool = True,
    static_frame: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda:0",
    audio_fpath: Optional[str] = None,
    self_reenact: bool = False,
    **kwargs,
):
    # to tensor
    verts = to_tensor(verts, device="cpu")
    idle_verts = to_tensor(idle_verts, device="cpu")
    n_frames = len(verts)

    # load reenact targets and render verts
    with load_renderer("ours", model_path, device, cache_model) as model:
        assert model is not None
        with load_data("ours", reenact_video, reenact_data_dir, "cpu", cache_reenact) as reenact_data:
            assert reenact_data is not None

            # compute metrics for self-reenact, we must use minimum n_frames
            if self_reenact:
                n_frames = min(n_frames, len(reenact_data["img"]))

            # iterate
            g_idx = 0
            out_dir = os.path.splitext(out_vpath)[0]
            if os.path.exists(out_dir):
                rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            for i in trange(0, n_frames, batch_size, desc="Rendering", leave=False):
                j = min(i + batch_size, n_frames)
                # get data batch
                batch = dict(
                    verts=verts[i:j],
                    idle_verts=idle_verts.unsqueeze(0),
                    rotat=fetch_batch(reenact_data["rot"], i, j, static_frame),
                    transl=fetch_batch(reenact_data["tsl"], i, j, static_frame),
                    cam=fetch_batch(reenact_data["cam"], i, j, static_frame),
                    background=fetch_batch(reenact_data["img"], i, j, static_frame),
                )
                batch = to_device(batch, device)  # cpu -> device
                # forward
                res = model(None, **batch)
                # process results and compute metrics if necessary
                for bi in range(len(res["nr_fake"])):
                    assert i + bi == g_idx
                    mask = res["mask"][bi].cpu().numpy().transpose(1, 2, 0)
                    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
                    mask = np.repeat(mask, 3, axis=-1)
                    fake = to_uint8_image(res["nr_fake"][bi], flip=True)
                    lossless_image(os.path.join(out_dir, f"{g_idx:06d}-fake.png"), fake)
                    lossless_image(os.path.join(out_dir, f"{g_idx:06d}-mask.png"), mask)
                    # next
                    g_idx += 1
                    # debug
                    # real = to_uint8_image(batch["background"][bi], flip=True)
                    # canvas = np.concatenate((real, fake, mask), axis=1)
                    # cv2.imshow("real | fake", canvas)
                    # cv2.waitKey(1)
            # to lossless video
            out_vmask = os.path.splitext(out_vpath)[0] + "-mask.mp4"
            lossless_video(os.path.join(out_dir, "%06d-fake.png"), out_vpath, apath=audio_fpath)
            lossless_video(os.path.join(out_dir, "%06d-mask.png"), out_vmask, apath=audio_fpath)
            # remove tmp dir
            rmtree(out_dir)
