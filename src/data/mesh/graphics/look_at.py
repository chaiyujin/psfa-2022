from typing import Any, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as onpt

ArrFlt = TypeVar("ArrFlt", onpt.NDArray[onp.float_], jax.Array)


def look_at(
    eye: ArrFlt,
    at: ArrFlt,
    up: ArrFlt,
) -> ArrFlt:
    assert eye.shape[-1] == 3
    assert at.shape[-1] == 3
    assert up.shape[-1] == 3
    # Which np?
    np: Any = jnp if all([isinstance(x, jax.Array) for x in [eye, at, up]]) else onp

    def _normalize(x: ArrFlt) -> ArrFlt:
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    def _dot(a: ArrFlt, b: ArrFlt) -> ArrFlt:
        return np.einsum("...k,...k->...", a, b)[..., None]

    f = _normalize(at - eye)
    s = _normalize(np.cross(f, up, axis=-1))
    u = np.cross(s, f, axis=-1)
    _0 = np.zeros_like(f[..., :1])
    _1 = np.ones_like(f[..., :1])

    mat = np.stack(
        [
            np.concatenate([s, -_dot(s, eye)], axis=-1),
            np.concatenate([u, -_dot(u, eye)], axis=-1),
            np.concatenate([-f, _dot(f, eye)], axis=-1),
            np.concatenate([_0, _0, _0, _1], axis=-1),
        ],
        axis=-2,
    )

    return mat
