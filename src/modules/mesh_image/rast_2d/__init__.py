import os

import numpy as np


def cython_rasterize_2d(pts, hw, triangle_indices):
    try:
        from . import rast
    except ImportError:
        os.system(f"cd {os.path.dirname(__file__)} && python3 setup.py build_ext --inplace")
        from . import rast

    # generate triangle masks
    rast_out = rast.rasterize_triangles_2d(pts, triangle_indices, *hw)
    return rast_out
