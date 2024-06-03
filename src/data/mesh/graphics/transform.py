from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as onpt

Tensor = Union[onpt.NDArray[Any], jax.Array]


def transform(
    mat: Tensor,
    pos: Tensor,
    normalize: bool = False,
    eps: Optional[float] = None,
    pad_value: float = 1.0,
):
    np: Any = jnp if isinstance(pos, jax.Array) else onp

    assert mat.ndim == 3 and mat.shape[-2] == 4 and mat.shape[-1] == 4, "Invalid matrix shape: {}".format(mat.shape)
    assert pos.ndim == 3 and pos.shape[-1] in [
        3,
        4,
    ], "Invalid position shape: {}".format(pos.shape)
    wanna_homo = pos.shape[-1] == 4
    # pad if necessary
    posw = np.pad(pos, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=pad_value) if pos.shape[-1] == 3 else pos
    # assert (posw[..., -1] == pad_value).all()

    # transform
    posw_clip = np.einsum("bvi,bji->bvj", posw, mat)
    # _x = np.einsum("bvi,bij->bvj", posw, np.transpose(mat, (0, 2, 1)))
    # assert np.all(_x == posw_clip)

    # normalize
    if normalize:
        # normalize
        denom = posw_clip[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).astype(denom.dtype)
            denom = denom_sign * np.clip(denom.abs(), a_min=eps)
        ret = posw_clip[..., :3] / denom
        if wanna_homo:
            # ret = F.pad(ret, (0, 1), "constant", 1)
            ret = np.concatenate((ret, posw_clip[..., 3:]), axix=-1)
    else:
        ret = posw_clip if wanna_homo else posw_clip[..., :3]
    # return
    return ret
