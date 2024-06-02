from copy import deepcopy

import torch.nn as nn

from . import ops
from .logging import get_logger

log = get_logger("ENGINE")


def motion(x):
    return x[:, 1:] - x[:, :-1]


def legacy_remove_padding(x, target_frames):
    if x is None:
        return x
    if x.shape[1] > target_frames:
        padlr = x.shape[1] - target_frames
        left = padlr // 2
        right = padlr - left
        s, e = left, x.shape[1] - right
        return x[:, s:e].contiguous()
    return x


def fold(x):
    if x is None:
        return x, None
    return x.contiguous().view(-1, *x.shape[2:]), x.shape[1]


def unfold(x, frm):
    if x is None or frm is None:
        return x
    if x.shape[0] == 1:
        return x.unsqueeze(1)
    return x.contiguous().view(x.shape[0] // frm, frm, *x.shape[1:])


def unfold_dict(data, frm):
    if frm is None:
        return
    for k in data:
        if isinstance(data[k], dict):
            unfold_dict(data[k], frm)
        else:
            data[k] = unfold(data[k], frm)
    return data


def fold_codes(code_dict):
    frm = None
    for key, val in code_dict.items():
        if val is None:
            continue
        # * check ndim
        ndim = 4 if key == "sh9" else 3
        if key not in ["shape", "tex"]:
            assert val.ndim == ndim or val.ndim == ndim - 1
            if val.ndim == ndim:
                if frm is not None:
                    assert frm == val.shape[1]
                frm = val.shape[1]

    if frm is None:
        return code_dict, None

    # * return new dict if fold
    ret = dict()
    for key, val in code_dict.items():
        if val is None:
            continue
        ret[key], _ = fold(val)
    return ret, frm


class _FoldUnfold(nn.Module):
    def __init__(self, layer, clone=True, freeze=False):
        super().__init__()
        self._layer = deepcopy(layer) if clone else layer
        if freeze:
            ops.freeze(self._layer)

    def forward(self, x):
        x, frm = fold(x)
        x = self._layer(x)
        if isinstance(x, tuple):
            x = x[0]
        return unfold(x, frm)


def FoldUnfold(layer, clone=True, freeze=False):
    if layer is None:
        return nn.Identity()
    else:
        return _FoldUnfold(layer, clone=clone, freeze=freeze)
