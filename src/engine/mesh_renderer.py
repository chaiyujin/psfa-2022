import cv2
import numpy as np
import torch

from assets import (
    get_selection_triangles,
    get_selection_vidx,
    get_vocaset_template_triangles,
    get_vocaset_template_vertices,
)
from src.data.mesh import mesh_renderer

# from src.engine.graphics import matrix
from src.engine.painter import color_mapping, put_colorbar

mesh_renderer.set_template(
    v=get_vocaset_template_vertices(),
    f=get_vocaset_template_triangles(),
)


def _verts_and_tris(verts):
    vidx = list(range(verts.shape[-2]))
    tris = get_vocaset_template_triangles()
    # vidx = get_selection_vidx("face")
    # tris = get_selection_triangles("face")
    return vidx, verts[vidx], tris


def _pvm(intrs=(2.0, 0.5, 0.5), transl=(0, 0, -0.5), euler=(0.0, 0.0, 0.0)):
    R = matrix.euler_rotate(torch.tensor(euler, dtype=torch.float32)).numpy()
    T = matrix.translate(torch.tensor(transl, dtype=torch.float32)).numpy()
    P = matrix.get_persp_projection(torch.tensor(intrs, dtype=torch.float32)).numpy()
    return np.matmul(P, np.matmul(T, R)).astype(np.float32)


# _pvm_front = _pvm((2.8, 0.5, 0.5), (+0.00, 0, -0.65), (0.0, 0.0, 0.0))
# _pvm_side = _pvm((2.8, 0.5, 0.5), (-0.05, 0, -0.65), (0.0, 1.4, 0.0))
# _pvm_half = _pvm((2.8, 0.5, 0.5), (-0.02, 0, -0.65), (0.0, 0.7, 0.0))
# _mat_eye = np.eye(4, dtype=np.float32)
# _common_settings = dict(dtype=np.float32, mat_view=_mat_eye, mat_model=_mat_eye)


# fmt: off
def render(verts, A=256, side=False, shading='smooth'):
    mesh_renderer.set_image_size(A, A)
    img = mesh_renderer.render(verts)
    return img[..., :3].astype(np.float32) / 255.0
    # return np.zeros((A, A, 3), dtype=np.uint8)
    _, verts, tris = _verts_and_tris(verts)
    if side is False:
        pvm = _pvm_front
    elif side is True:
        pvm = _pvm_side
    elif side == "45":
        pvm = _pvm_half
    img = mesh_viewer.render(verts, f=tris, image_size=(A, A), shading=shading, mat_proj=pvm, **_common_settings)
    return img[..., :3]


def render_heatmap(verts, vert_values, A=256, vmin=None, vmax=None, unit='', scale=1.0):
    vidx, verts, tris = _verts_and_tris(verts)
    vert_values = vert_values[..., vidx]
    vmin = vmin or vert_values.min()
    vmax = vmax or vert_values.max()

    vert_values = vert_values * scale
    vmin, vmax = vmin * scale, vmax * scale

    # values -> colors
    vert_colors = color_mapping(vert_values, vmin=vmin, vmax=vmax, cmap="jet")

    img = mesh_viewer.render(
        verts, f=tris, image_size=(A, A), mat_proj=_pvm_front,
        vertex_colors=vert_colors, **_common_settings,
        shading=None,
    )[..., :3]
    img = put_colorbar(
        img, cmap="jet", vmin=vmin, vmax=vmax, at="top",
        fmt="{:.0f}", unit=unit, font_size=int(12.0 * A / 256.0),
    )
    return img
# fmt: on
