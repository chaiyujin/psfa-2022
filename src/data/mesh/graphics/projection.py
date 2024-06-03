from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as onpt

Tensor = Union[onpt.NDArray[Any], jax.Array]


def _to_tensor(intrinsics: Any, dtype: Any) -> Tensor:
    assert isinstance(intrinsics, (onp.ndarray, jax.Array))
    return intrinsics


def _parse_intrinsics(intrinsics: Tensor, aspect: Optional[float]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Which np?
    np: Any = jnp if isinstance(intrinsics, jax.Array) else onp

    if intrinsics.shape[-1] == 3:
        fx, ppx, ppy = np.split(intrinsics, 3, axis=-1)  # type: ignore
        fy = fx if aspect is None else (fx / aspect)  # type: ignore
    else:
        fx, fy, ppx, ppy = np.split(intrinsics, 4, axis=-1)  # type: ignore
        if aspect is not None:
            fy = fx / aspect  # type: ignore
    return fx, fy, ppx, ppy  # type: ignore


def get_persp_projection(
    intrinsics: Any,
    zn: float = 0.1,
    zf: float = 100.0,
    aspect: Optional[float] = None,
):
    intrinsics = _to_tensor(intrinsics, dtype="float32")
    assert intrinsics.shape[-1] in [3, 4]
    dtype: Any = intrinsics.dtype

    # Which np?
    np: Any = jnp if isinstance(intrinsics, jax.Array) else onp

    # get parameters
    fx, fy, ppx, ppy = _parse_intrinsics(intrinsics, aspect)

    a = -(zf + zn) / (zf - zn)
    b = -2.0 * zf * zn / (zf - zn)

    # fill constant
    _a = np.full_like(fx, a, dtype=dtype)
    _b = np.full_like(fx, b, dtype=dtype)
    _c = np.full_like(fx, -1, dtype=dtype)
    _0 = np.full_like(fx, 0, dtype=dtype)
    _1 = np.full_like(fx, 1, dtype=dtype)

    # focus
    _x = fx * 2.0
    _y = fy * 2.0
    # center
    cx = _1 - ppx * 2.0
    cy = ppy * 2.0 - _1

    # matrix
    mat = np.stack(
        [
            np.concatenate([_x, _0, cx, _0], axis=-1),
            np.concatenate([_0, _y, cy, _0], axis=-1),
            np.concatenate([_0, _0, _a, _b], axis=-1),
            np.concatenate([_0, _0, _c, _0], axis=-1),
        ],
        axis=-2,
    )
    return mat


def get_ortho_projection(
    intrinsics: Any,
    zn: float = 0.01,
    zf: float = 100.0,
    aspect: Optional[float] = None,
):
    intrinsics = _to_tensor(intrinsics, dtype="float32")
    assert intrinsics.shape[-1] in [3, 4]
    dtype: Any = intrinsics.dtype

    # Which np?
    np: Any = jnp if isinstance(intrinsics, jax.Array) else onp

    # get parameters
    fx, fy, ppx, ppy = _parse_intrinsics(intrinsics, aspect)

    # constants
    a = -2.0 / (zf - zn)
    b = -1.0 * (zf + zn) / (zf - zn)

    # fill constant
    _a = np.full_like(fx, a, dtype=dtype)
    _b = np.full_like(fx, b, dtype=dtype)
    _0 = np.full_like(fx, 0, dtype=dtype)
    _1 = np.full_like(fx, 1, dtype=dtype)

    # focus
    _x = fx * 2.0
    _y = fy * 2.0
    # center
    cx = ppx * 2.0 - 1.0
    cy = 1.0 - ppy * 2.0

    # matrix
    mat = np.stack(
        [
            np.concatenate([_x, _0, _0, cx], axis=-1),
            np.concatenate([_0, _y, _0, cy], axis=-1),
            np.concatenate([_0, _0, _a, _b], axis=-1),
            np.concatenate([_0, _0, _0, _1], axis=-1),
        ],
        axis=-2,
    )
    return mat
