import heapq
import math
import os

import numpy as np
import scipy.sparse as sp

from . import utils
from .aabb_tree import AabbTree
from .mesh import Mesh


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])  # noqa

    nearest_faces, nearest_parts, nearest_vertices = AabbTree(source.v, source.f).nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i : 3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v  # noqa

        rows[3 * i : 3 * i + 3] = i * np.ones(3)
        cols[3 * i : 3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i : 3 * i + 3] = np.linalg.lstsq(A, nearest_v, rcond=-1)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]], source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i], rcond=-1)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh_v, mesh_f, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception("Need either factor or n_verts_desired.")

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh_v) * factor)

    Qv = utils.vertex_quadrics(mesh_v, mesh_f)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = utils.get_vertices_per_edge(mesh_v, mesh_f)
    # vert_adj = sp.lil_matrix((len(mesh_v), len(mesh_v)))
    # for f_idx in range(len(mesh_f)):
    #     vert_adj[mesh_f[f_idx], mesh_f[f_idx]] = 1

    vert_adj = sp.csc_matrix(
        (vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh_v), len(mesh_v)),
    )
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            "destroy_c_cost": destroy_c_cost,
            "destroy_r_cost": destroy_r_cost,
            "collapse_cost": min([destroy_c_cost, destroy_r_cost]),
            "Qsum": Qsum,
        }
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)["collapse_cost"]
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh_v)
    faces = mesh_f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh_v)
        if cost["collapse_cost"] > e[0]:
            heapq.heappush(queue, (cost["collapse_cost"], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost["destroy_c_cost"] < cost["destroy_r_cost"]:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost["Qsum"]
            Qv[c, :, :] = cost["Qsum"]

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = len(np.unique(faces.flatten()))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh_v))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return (new_faces, mtx)


def generate_sampling_information(mesh_v, mesh_f, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: csc_matrix Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
       F: a list of faces
    """

    M, A, E, D, U = [], [], [], [], []
    factors = map(lambda x: 1.0 / x, factors)

    def append_mesh(v, f):
        a = utils.get_vert_connectivity(v, f).astype("float32")
        e = np.asarray([a.tocoo().row, a.tocoo().col])
        M.append(Mesh(v, f, e=e))
        A.append(a)
        E.append(e)

    # original mesh
    append_mesh(mesh_v, mesh_f)

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1].v, M[-1].f, factor=factor)
        ds_v = ds_D.dot(M[-1].v)
        # mesh mesh
        append_mesh(ds_v, ds_f)
        # append sampling matrix
        U.append(setup_deformation_transfer(M[-1], M[-2]).astype("float32"))
        D.append(ds_D.astype("float32"))

    return dict(M=M, A=A, E=E, D=D, U=U)
