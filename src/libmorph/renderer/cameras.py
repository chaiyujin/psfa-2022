import numpy as np
import torch

from ..utils.matrix import get_ortho_projection, get_persp_projection, transform
from ..utils.tensor_props import TensorProperties


class Cameras(TensorProperties):
    def __init__(self, intrinsics):
        # view matrix
        view = np.eye(4, dtype=np.float32)[None, ...]
        view[:, 2, 3] = -1.0
        super().__init__(
            intrinsics=intrinsics,
            view_matrix=view,
            znear=0.01,
            zfar=100.0,
            __shared_keys__=("view_matrix", "znear", "zfar"),
        )

    def get_world_to_view_matrix(self, **kwargs):
        return self.view_matrix

    def get_projection_matrix(self, **kwargs):
        raise NotImplementedError()

    def _get_pv(self, bsz, **kwargs):
        w2v_mat = self.get_world_to_view_matrix(**kwargs)
        if w2v_mat.shape[0] != bsz:
            assert w2v_mat.shape[0] == 1
            w2v_mat = w2v_mat.expand(bsz, -1, -1)

        prj_mat = self.get_projection_matrix(**kwargs)
        if prj_mat.shape[0] != bsz:
            assert prj_mat.shape[0] == 1
            prj_mat = prj_mat.expand(bsz, -1, -1)

        return torch.bmm(prj_mat, w2v_mat)

    def _flip_y(self, vertices):
        device = vertices.device
        if vertices.shape[-1] == 4:
            vertices = vertices * torch.tensor([1, -1, 1, 1], device=device)
        else:
            assert vertices.shape[-1] == 3
            vertices = vertices * torch.tensor([1, -1, 1], device=device)
        return vertices

    def transform(self, vertices, eps=None, **kwargs):
        bsz = vertices.shape[0]
        matrix = self._get_pv(bsz, **kwargs)
        norm_verts = transform(matrix, vertices, normalize=True, eps=eps)
        # ! flip y axis: we follow OpenGL, so flip y axis at last
        return self._flip_y(norm_verts)


class PerspectiveCameras(Cameras):
    def __init__(self, camera_aspect=None):
        super().__init__([[3.0, 0.5, 0.5]])
        self.aspect = camera_aspect
        self.camera_type = "perspective"

    def get_projection_matrix(self, **kwargs):
        self.update_attr("intrinsics", **kwargs)
        self.update_attr("znear", **kwargs)
        self.update_attr("zfar", **kwargs)
        return get_persp_projection(self.intrinsics, self.znear, self.zfar, self.aspect)


class OrthographicCameras(Cameras):
    def __init__(self, camera_aspect=None):
        super().__init__([[3.0, 0.5, 0.5]])
        self.aspect = camera_aspect
        self.camera_type = "orthographic"

    def get_projection_matrix(self, **kwargs):
        self.update_attr("intrinsics", **kwargs)
        self.update_attr("znear", **kwargs)
        self.update_attr("zfar", **kwargs)
        return get_ortho_projection(self.intrinsics, self.znear, self.zfar, self.aspect)
