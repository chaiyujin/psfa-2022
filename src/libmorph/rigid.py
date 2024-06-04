import numpy as np
import torch
import torch.nn as nn

from .utils.matrix import euler_rotate
from .utils.tensor_props import TensorProperties


class Rigid(TensorProperties):
    def __init__(self):
        super().__init__(
            rotat=np.zeros((1, 3), dtype=np.float32),
            transl=np.zeros((1, 3), dtype=np.float32),
        )

    def forward(self, vertices, rotat=None, transl=None):
        self.update_attr("rotat", rotat=rotat)
        self.update_attr("transl", transl=transl)
        assert vertices.ndim == 3
        assert self.rotat.ndim == 2
        assert self.transl.ndim == 2

        # * rotation
        rot_mat = euler_rotate(self.rotat)[:, :3, :3]
        if rot_mat.shape[0] == 1 and rot_mat.shape != vertices.shape[0]:
            rot_mat = rot_mat.expand(vertices.shape[0], -1, -1)
        vertices = torch.bmm(vertices, rot_mat.permute(0, 2, 1))

        # * translation
        vertices = vertices + self.transl.unsqueeze(1)

        return vertices
