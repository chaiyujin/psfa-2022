from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as onpt

Tensor = Union[onpt.NDArray[Any], jax.Array]


def translate(vec: Any, dtype: Any = None):
    np: Any = jnp if isinstance(vec, jax.Array) else onp
    assert vec.shape[-1] == 3

    # chunk components
    _x, _y, _z = np.split(vec, 3, axis=-1)
    # stack the matrix
    _0 = np.zeros_like(_x)
    _1 = np.ones_like(_x)
    mat = np.stack(
        [
            np.concatenate([_1, _0, _0, _x], axis=-1),
            np.concatenate([_0, _1, _0, _y], axis=-1),
            np.concatenate([_0, _0, _1, _z], axis=-1),
            np.concatenate([_0, _0, _0, _1], axis=-1),
        ],
        axis=-2,
    )
    return mat


def _euler_near_zero_approx(vec: Tensor):
    np: Any = jnp if isinstance(vec, jax.Array) else onp

    _x, _y, _z = np.split(vec, 3, axis=-1)
    nx = _x * -1.0
    ny = _y * -1.0
    nz = _z * -1.0
    _0 = np.zeros_like(_x)
    _1 = np.ones_like(_x)

    mat = np.stack(
        [
            np.concatenate([_1, nz, _y, _0], axis=-1),
            np.concatenate([_z, _1, nx, _0], axis=-1),
            np.concatenate([ny, _x, _1, _0], axis=-1),
            np.concatenate([_0, _0, _0, _1], axis=-1),
        ],
        axis=-2,
    )
    return mat


def euler_rotate(vec: Any, eps: float = 1e-8, dtype: Any = None):
    np: Any = jnp if isinstance(vec, jax.Array) else onp
    assert vec.shape[-1] == 3

    # get angle
    angle: Tensor = np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)  # type: ignore

    # 1. near zero approx
    near_zero_mask = angle <= eps
    mat_approx = _euler_near_zero_approx(vec)

    # 2. to angle-axis
    angle = np.where(near_zero_mask, np.full_like(angle, eps), angle)
    axis = vec / angle
    # angle-axis to quat
    angle = angle * 0.5
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.concatenate([v_cos, v_sin * axis], axis=-1)
    # quat to mat
    mat_quat = quat_rotate(quat, is_normalized=True)

    # 3. merge
    mat = np.where(near_zero_mask[..., np.newaxis], mat_approx, mat_quat)

    return mat


def euler2quat(vec: Any, eps: float = 1e-8, dtype: Any = None):
    np: Any = jnp if isinstance(vec, jax.Array) else onp
    assert vec.shape[-1] == 3

    # to angle-axis
    angle: Tensor = np.linalg.norm(vec + eps, ord=2, axis=-1, keepdims=True)  # type: ignore
    axis = vec / angle

    # angle-axis to quat
    angle = angle * 0.5
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.concatenate([v_cos, v_sin * axis], axis=-1)

    return quat


def quat_rotate(quat: Any, is_normalized: bool = False, dtype: Any = None):
    np: Any = jnp if isinstance(quat, jax.Array) else onp
    assert quat.shape[-1] == 4

    norm_quat = quat
    # not sure quat is normalized or not
    if not is_normalized:
        norm_quat = norm_quat / np.linalg.norm(norm_quat, ord=2, axis=-1, keepdims=True)

    w, x, y, z = np.split(norm_quat, 4, axis=-1)

    w2, x2, y2, z2 = np.power(w, 2), np.power(x, 2), np.power(y, 2), np.power(z, 2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    _0 = np.zeros_like(w2)
    _1 = np.ones_like(w2)

    # fmt: off
    mat = np.stack([
        np.concatenate([w2 + x2 - y2 - z2,  2 * xy - 2 * wz,    2 * wy + 2 * xz,    _0], axis=-1),
        np.concatenate([2 * wz + 2 * xy,    w2 - x2 + y2 - z2,  2 * yz - 2 * wx,    _0], axis=-1),
        np.concatenate([2 * xz - 2 * wy,    2 * wx + 2 * yz,    w2 - x2 - y2 + z2,  _0], axis=-1),
        np.concatenate([_0,                 _0,                 _0,                 _1], axis=-1),
    ], axis=-2)
    # fmt: on

    return mat
