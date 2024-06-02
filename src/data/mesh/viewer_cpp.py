import os

import cv2
import numpy as np

from src.cpp.src.mesh_viewer.build import mesh_viewer
from src.cpp.src.mesh_viewer.build.mesh_viewer import (
    get_contour,
    render_verts,
    set_background_image,
    set_background_mode,
    set_clear_color,
    set_cm_values,
    set_contour_template,
    set_draw_cm_flag,
    set_draw_cm_with_shading_flag,
    set_resolution,
    set_shading_mode,
    set_texture,
    set_verts,
)

from .io import load_mesh

_root = os.path.join(os.path.abspath(os.path.dirname(__file__)))
# set voca template
_template_verts, _template_faces = None, None


def set_template(filename):
    global _template_verts
    global _template_faces
    assert isinstance(filename, str)
    assert os.path.splitext(filename)[1] == ".obj"

    mesh_viewer.set_template(filename)
    _template_verts, _template_faces, _ = load_mesh(filename, flatten=True)


def template_verts():
    if _template_verts is None:
        return None
    return np.reshape(_template_verts, (-1, 3))


def template_faces():
    if _template_faces is None:
        return None
    return np.reshape(_template_faces, (-1, 3))


def render_mesh(verts: np.ndarray, faces: np.ndarray = None, image_size: tuple = (512, 512), dtype=np.uint8, **kwargs):
    assert (
        _template_verts is not None and _template_faces is not None
    ), "Template is not set yet! please call set_template()"
    # check same
    if faces is not None:
        assert np.all(faces.flatten() == _template_faces.flatten())
    verts = verts.flatten(order="C").astype(np.float32)
    assert len(verts) == len(_template_verts), "given verts length should be {}! not {}".format(
        len(_template_verts), len(verts)
    )

    # FIXME: not good in cpp
    # mesh_viewer.set_resolution(*image_size)

    img = mesh_viewer.render_verts(verts, film_size=image_size, **kwargs)
    if image_size != (img.shape[1], img.shape[0]):
        img = cv2.resize(img, image_size)

    # return in correct dtype and range
    if dtype == np.uint8:
        return img
    elif dtype in [np.float32, np.float64]:
        return img.astype(dtype) / 255.0
    else:
        raise TypeError("cannot render image with dtype: {}".format(dtype))
