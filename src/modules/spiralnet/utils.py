from functools import lru_cache

import numpy as np
import openmesh as om
import torch
from sklearn.neighbors import KDTree

from src.data.mesh.numpy import Mesh, generate_sampling_information

# -------------------------------------------------------------------------------------------------------------------- #
#                                                         Utils                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def _to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data),
        torch.Size(spmat.tocoo().shape),
    )


def _preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = np.asarray(extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def get_spiral_indices(M, seq_lengths, dilations):
    return [_preprocess_spiral(M[idx].f, seq_lengths[idx], M[idx].v, dilations[idx]) for idx in range(len(M) - 1)]


@lru_cache(maxsize=1)
def preprocess_template(template_path, seq_lengths, dilations, ds_factors):
    assert template_path is not None
    assert len(seq_lengths) == len(dilations)
    assert len(seq_lengths) == len(ds_factors)
    mesh = Mesh.LoadFile(template_path)
    info = generate_sampling_information(mesh.v, mesh.f, ds_factors)
    M, D, U = info["M"], info["D"], info["U"]
    spiral_indices_list = get_spiral_indices(M, seq_lengths, dilations)
    down_transform_list = [_to_sparse(x) for x in D]
    up_transform_list = [_to_sparse(x) for x in U]
    return spiral_indices_list, down_transform_list, up_transform_list


# -------------------------------------------------------------------------------------------------------------------- #
#                                                    generate spiral                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return idx not in last_ring and idx not in other and idx not in res

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric="euclidean")
            spiral = kdt.query(
                np.expand_dims(mesh.points()[spiral[0]], axis=0),
                k=seq_length * dilation,
                return_distance=False,
            ).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[: seq_length * dilation][::dilation])
    return spirals
