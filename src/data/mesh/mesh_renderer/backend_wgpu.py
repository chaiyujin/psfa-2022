from __future__ import annotations

from logging import getLogger
from typing import Any, List, Optional, Union

import jax
import numpy as onp
import numpy.typing as onpt

from ..graphics import face_normals, look_at, projection, vertex_normals
from .renderer import MeshRenderer

LOG = getLogger("[mesh_renderer|wgpu]")

Tensor = Union[onpt.NDArray[Any], jax.Array]
_default_size = (512, 512)
_g_normal_mode: str = "smooth"
_g_v: Optional[onpt.NDArray[onp.float32]] = None
_g_f: Optional[onpt.NDArray[onp.int32]] = None
_g_r: Optional[MeshRenderer] = None
_default_mat_view = look_at(
    eye=onp.array([0.0, 0.0, 1.0]), at=onp.array([0.0, 0.0, 0.0]), up=onp.array([0.0, 1.0, 0.0])
)
_default_mat_proj = projection.get_persp_projection(onp.array([4.0, 0.5, 0.5], dtype="float32"))


def _to_numpy(x: Tensor, dtype: Any = "float32") -> onpt.NDArray[Any]:
    return jax.device_get(x).astype(dtype) if isinstance(x, jax.Array) else x


def set_template(v: Tensor, f: Tensor, mode: str = "smooth"):
    global _g_f, _g_v, _g_normal_mode, _g_r
    if (_g_f is None) or (_g_f.shape[0] != f.shape[0]):
        _g_v = _to_numpy(v, "float32")
        _g_f = _to_numpy(f, "int32")
        _g_normal_mode = mode

        n_tris = _g_f.shape[0]
        _g_r = MeshRenderer.create_offscreen(*_default_size)
        _g_r.initialize(n_tris * 3, n_tris, texture_size=512)
        _g_r.update_indices(onp.arange(n_tris * 3, dtype="uint32"))


def set_image_size(w: int, h: int):
    if _g_r is None:
        assert _g_r is None

        global _default_size
        _default_size = (w, h)
    elif _default_size != (w, h):
        LOG.warning(f"Ignore image_size: {w}x{h}")


def render(
    position: Tensor,
    mat_model: Tensor | None = None,
    mat_view: Tensor | None = None,
    mat_proj: Tensor | None = None,
    vert_rgb: Tensor | None = None,
    lighting: bool = True,
) -> onpt.NDArray[onp.uint8]:
    # Check initialized.
    assert _g_v is not None and _g_f is not None, "Please call `set_template(v, f)` first!"
    assert _g_r is not None
    r = _g_r

    # Check inputs.
    assert position.ndim == 2 and position.shape[-1] == 3, f"Invalid shape: {position.shape}"
    pos = _to_numpy(position)
    xyz = pos[_g_f.flatten()]
    normal = face_normals(pos[None], _g_f, flat=_g_normal_mode == "flat").reshape(-1, 3)
    r.update_vertices(xyz, part="xyz")
    r.update_vertices(normal, part="normal")
    # (Optional) vert_rgb.
    if vert_rgb is not None:
        np_rgb = _to_numpy(vert_rgb)[_g_f.flatten()]
        r.update_vertices(np_rgb, part="rgb")

    # Check transform matrices.
    assert mat_model is None or mat_model.shape == (4, 4), f"Invalid shape: {mat_model.shape}"
    assert mat_view is None or mat_view.shape == (4, 4), f"Invalid shape: {mat_view.shape}"
    assert mat_proj is None or mat_proj.shape == (4, 4), f"Invalid shape: {mat_proj.shape}"
    r.update_transform(
        mat_model=_to_numpy(mat_model) if mat_model is not None else None,
        mat_view=_to_numpy(mat_view) if mat_view is not None else _default_mat_view,
        mat_proj=_to_numpy(mat_proj) if mat_proj is not None else _default_mat_proj,
        is_opengl=True,
    )

    r.toggle("shading", lighting)

    return r.draw()
