""" Wrap opencv api: the default color format in our project is RGB """

import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision as tv
from PIL import Image


def _X_is_running():
    try:
        from subprocess import PIPE, Popen

        p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
        p.communicate()
        return p.returncode == 0
    except:
        return True


IS_X_AVAILABLE = _X_is_running()


def imread(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(filename, img, makedirs: bool = False):
    if makedirs and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if img.dtype not in ["uint8", np.uint8]:
        img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, np.clip(img, 0, 255))


def imshow(winname, img):
    if IS_X_AVAILABLE:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(winname, img)


def waitKey(msec=0):
    if IS_X_AVAILABLE:
        return cv2.waitKey(msec)


def resize(
    img,
    dsize: Optional[Tuple[int, int]] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    interpolation=cv2.INTER_LINEAR,
):
    if fx is not None or fy is not None:
        assert dsize is None, "You cannot set dsize and fx,fy at same time!"
        dsize = (int(img.shape[1] * (fx or 1.0)), int(img.shape[0] * (fy or 1.0)))
    img = cv2.resize(img, dsize, interpolation=interpolation)
    if img.ndim == 2:
        img = img[..., None]
    return img


def load_th(
    image_path: str, transforms: Optional[List[Callable]] = None, device: str = "cpu", vrange=None
) -> torch.Tensor:
    # load by Pillow
    img = Image.open(image_path)
    # torchvision transforms
    if transforms is None:
        transforms = []
    trans_fn = tv.transforms.Compose(transforms + [tv.transforms.ToTensor()])  # to tensor at last
    img = trans_fn(img)
    # 0~1 -> vrange
    if vrange is not None:
        assert len(vrange) == 2
        vmin, vmax = vrange
        img = img * (vmax - vmin) + vmin
    # move to device
    return img.to(device)


def may_CHW2HWC(img):
    if 1 <= img.shape[-3] <= 4:
        ndim = img.ndim
        new_shape = list(range(ndim - 3)) + [ndim - 2, ndim - 1, ndim - 3]
        if torch.is_tensor(img):
            img = img.permute(new_shape)
        else:
            assert isinstance(img, np.ndarray)
            img = np.transpose(img, new_shape)
    assert 1 <= img.shape[-1] <= 4
    return img
