from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Colors:
    # fmt: off
    red   : Tuple[int, int, int] = (237, 68, 66)
    green : Tuple[int, int, int] = (173, 244, 186)
    blue  : Tuple[int, int, int] = (143, 228, 255)
    white : Tuple[int, int, int] = (255, 255, 255)
    # fmt: on


class Painter(object):

    # Class fancy colors
    colors = Colors()

    @classmethod
    def get(cls, data, key, bi, fi) -> Optional[np.ndarray]:
        idx = key.find(".")
        if idx >= 0:
            k, key = key[:idx], key[idx + 1 :]
            if not isinstance(data.get(k), dict):
                return None
            else:
                return cls.get(data[k], key, bi, fi)
        else:
            t = data.get(key)
            if t is None:
                return None
            t = t[bi, fi].detach().cpu().numpy()
            return t

    @classmethod
    def normalize_img(cls, img):
        if img is None:
            return None
        return (img + 1.0) / 2.0

    @classmethod
    def draw_landmarks(cls, v_frame, lmks_pred, lmks_true, lmks_weights):
        if isinstance(lmks_weights, float):
            lmks_weights = None

        # speically for full target and landmarks
        lmks_canvas = np.clip(np.ascontiguousarray(np.copy(v_frame)), 0, 1)

        def _draw_pts(img, pts, radius, color, thickness=1):
            if pts is None:
                return
            color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            for i, p in enumerate(pts):
                # ignore not used point
                if lmks_weights is not None and lmks_weights[i] <= 0:
                    continue
                center = (int((p[0] + 1) / 2 * img.shape[1]), int((p[1] + 1) / 2 * img.shape[0]))
                cv2.circle(img, center, radius, color, thickness=thickness)

        _draw_pts(lmks_canvas, lmks_true, 2, cls.colors.green, -1)
        _draw_pts(lmks_canvas, lmks_pred, 2, cls.colors.red, -1)

        return lmks_canvas

    @classmethod
    def masked(cls, img, mask, fill_val=0):
        return np.where(mask, img, np.full_like(img, fill_val))

    @classmethod
    def may_CHW2HWC(cls, img):
        if 1 <= img.shape[-3] <= 4:
            ndim = img.ndim
            new_shape = list(img.shape[:-3]) + [ndim - 2, ndim - 1, ndim - 3]
            img = np.transpose(img, new_shape)
        assert img.shape[-1] in [1, 2, 3, 4]
        return img


class ClassMethod(object):
    "Emulate PyClassMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)

        def newfunc(*args, **kwargs):
            return self.f(klass, *args, **kwargs)

        newfunc.__name__ = self.f.__name__
        return newfunc


def register_painter(fn):
    setattr(Painter, fn.__name__, ClassMethod(fn))
    return fn


def parse_painter(fn_or_name):
    if callable(fn_or_name):
        fn = fn_or_name
    else:
        fn = getattr(Painter, fn_or_name)
    assert callable(fn)
    return fn


def get_painters(config):
    draw_fns = config.get("painter")
    if draw_fns is None:
        return []
    if isinstance(draw_fns, str):
        draw_fns = [draw_fns]
    return [parse_painter(x) for x in draw_fns]
