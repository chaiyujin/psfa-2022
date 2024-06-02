from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def concat_attrs(*args: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    d_list = [x.shape[-1] for x in args]
    return torch.cat(args, dim=-1), d_list


def chunk_attrs(tensor: torch.Tensor, d_list: List[int]) -> List[torch.Tensor]:
    assert sum(d_list) == tensor.shape[-1]
    ret, s = [], 0
    for length in d_list:
        ret.append(tensor[..., s : s + length])
        s += length
    return ret


def merge_attr_dict(**kwargs) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
    tensors, kd_list = [], []
    for k in sorted(list(kwargs.keys())):
        v = kwargs[k]
        tensors.append(v)
        kd_list.append((k, v.shape[-1]))
    return torch.cat(tensors, dim=-1), kd_list


def split_attr_dict(tensor: torch.Tensor, kd_list: List[Tuple[str, int]], key_prefix="") -> Dict[str, torch.Tensor]:
    assert sum(x[1] for x in kd_list) == tensor.shape[-1]
    ret, s = dict(), 0
    for k, d in kd_list:
        v = tensor[..., s : s + d]
        s += d
        ret[key_prefix + k] = v
    return ret


def vert2face_attrs(vertex_attrs: Tensor, faces: Tensor) -> Tensor:
    """
    :param vertex_attrs: [batch size, number of vertices, C]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertex_attrs.shape[0]:
        assert faces.shape[0] == 1
        faces = faces.expand(vertex_attrs.shape[0], -1, -1)
    assert vertex_attrs.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertex_attrs.shape[0] == faces.shape[0]
    # assert (vertices.shape[2] == 3)
    assert faces.shape[2] == 3

    bs, nv = vertex_attrs.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertex_attrs.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    # pytorch only supports long and byte tensors for indexing
    return vertex_attrs.view(bs * nv, vertex_attrs.shape[-1])[faces.long()]


def vertex_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertices.shape[0]:
        assert faces.shape[0] == 1
        faces = faces.expand(vertices.shape[0], -1, -1)
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    # pytorch only supports long and byte tensors for indexing
    vertices_faces = vertices.view(bs * nv, 3)[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.view(bs, nv, 3)
    return normals


def face_normals(vertices: Tensor, faces: Tensor, flat: bool = False) -> Tensor:
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :param flat: bool
    :return: [batch size, number of faces, 3, 3]
    """
    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertices.shape[0]:
        assert faces.shape[0] == 1
        faces = faces.expand(vertices.shape[0], -1, -1)
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    if not flat:
        v_normals = vertex_normals(vertices, faces)
        return vert2face_attrs(v_normals, faces)
    else:
        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
        vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]
        vertices_faces = vertices_faces.view(-1, 3, 3)

        nx = torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0])
        nx = F.normalize(nx, eps=1e-6, dim=1)

        return nx.unsqueeze(-2).repeat(1, 3, 1).view(bs, nf, 3, 3)
