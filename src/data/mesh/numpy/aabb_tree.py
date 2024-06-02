import os
import sys

import numpy as np

# spatialsearch, build if failed to import
try:
    from .psbody import spatialsearch
except ImportError:
    # * build
    pwd = os.path.abspath(os.getcwd())
    pkdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "psbody")
    os.system(f"cd {pkdir} && {sys.executable} setup.py build_ext --inplace && cd {pwd}")
    from .psbody import spatialsearch


class AabbTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree"""

    def __init__(self, v, f):
        verts = v.astype(np.float64)
        faces = f.astype(np.uint32)
        self.cpp_handle = spatialsearch.aabbtree_compute(verts.copy(order="C"), faces.copy(order="C"))

    def nearest(self, v_samples, nearest_part=False):
        """nearest_part tells you whether the closest point in triangle abc is in the interior (0),
        on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
        """
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(
            self.cpp_handle, np.array(v_samples, dtype=np.float64, order="C")
        )
        return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)

    def nearest_alongnormal(self, points, normals):
        distances, f_idxs, v = spatialsearch.aabbtree_nearest_alongnormal(
            self.cpp_handle, points.astype(np.float64), normals.astype(np.float64)
        )
        return (distances, f_idxs, v)
