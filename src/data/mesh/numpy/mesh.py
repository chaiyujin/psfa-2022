import os
from typing import Optional

import numpy as np

from .. import io
from . import utils


class Mesh(object):
    def __init__(self, v: np.ndarray, f: np.ndarray, e: Optional[np.ndarray] = None):
        self.v = v
        self.f = f
        self.e = e
        # check as np.ndarray
        for k, v in self.__dict__.items():
            assert v is None or isinstance(v, np.ndarray)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                      File IO                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    def load(self, filename: str):
        v, f, _ = io.load_mesh(filename, dtype=np.float64)
        self.v = v
        self.f = f
        # adj and edges
        a = utils.get_vert_connectivity(v, f)
        self.e = np.asarray([a.tocoo().row, a.tocoo().col])

    def save(self, filename: str):
        assert self.v.ndim == 2 and self.v.shape[-1] == 3
        assert self.f.ndim == 2 and self.f.shape[-1] == 3
        ext = os.path.splitext(filename)[1]
        if ext == ".obj":
            io.save_obj(filename, self.v, self.f)
        elif ext == ".ply":
            io.save_ply(filename, self.v, self.f)
        else:
            raise ValueError("Unknown extension for mesh file: '{}'. Only .obj|.ply are supported.".format(filename))

    @staticmethod
    def LoadFile(filename):
        m = Mesh(None, None)
        m.load(filename)
        return m

    @staticmethod
    def SaveFile(filename: str, v, f):
        m = Mesh(v, f)
        m.save(filename)
