from typing import Any, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as onpt

ArrFlt = TypeVar("ArrFlt", onpt.NDArray[onp.float32], jax.Array)
ArrInt = TypeVar("ArrInt", onpt.NDArray[onp.int32], jax.Array)


def vert2face_attrs(vertex_attrs: ArrFlt, faces: ArrInt) -> ArrFlt:
    """
    :param vertex_attrs: [batch size, number of vertices, C]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    np: Any = jnp if isinstance(vertex_attrs, jax.Array) else onp

    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertex_attrs.shape[0]:
        assert faces.shape[0] == 1
        faces = np.broadcast_to(faces, (vertex_attrs.shape[0], faces.shape[1], faces.shape[2]))
    assert vertex_attrs.ndim == 3
    assert faces.ndim == 3
    assert vertex_attrs.shape[0] == faces.shape[0]
    # assert (vertices.shape[2] == 3)
    assert faces.shape[2] == 3

    bs, nv, na = vertex_attrs.shape
    bs, _ = faces.shape[:2]
    faces = faces + (np.arange(bs, dtype=faces.dtype) * nv)[:, None, None]
    return np.reshape(vertex_attrs, (bs * nv, na))[faces]


def vertex_normals(vertices: ArrFlt, faces: ArrInt) -> ArrFlt:
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    np: Any = jnp if isinstance(vertices, jax.Array) else onp

    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertices.shape[0]:
        assert faces.shape[0] == 1
        faces = np.broadcast_to(faces, (vertices.shape[0], faces.shape[1], faces.shape[2]))
    assert vertices.ndim == 3
    assert faces.ndim == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, _ = faces.shape[:2]
    normals = np.zeros((bs * nv, 3), dtype=vertices.dtype)

    faces = faces + (np.arange(bs, dtype=faces.dtype) * nv)[:, None, None]  # expanded faces
    vertices_faces = np.reshape(vertices, (bs * nv, 3))[faces]

    faces = np.reshape(faces, (-1, 3))
    vertices_faces = np.reshape(vertices_faces, (-1, 3, 3))

    n1 = np.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
    )
    n2 = np.cross(
        vertices_faces[:, 0] - vertices_faces[:, 2],
        vertices_faces[:, 1] - vertices_faces[:, 2],
    )
    n0 = np.cross(
        vertices_faces[:, 1] - vertices_faces[:, 0],
        vertices_faces[:, 2] - vertices_faces[:, 0],
    )
    if np == onp:
        np.add.at(normals, faces[:, 1], n1)
        np.add.at(normals, faces[:, 2], n2)
        np.add.at(normals, faces[:, 0], n0)
    else:
        normals = normals.at[faces[:, 1]].add(n1)
        normals = normals.at[faces[:, 2]].add(n2)
        normals = normals.at[faces[:, 0]].add(n0)

    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
    return np.reshape(normals, (bs, nv, 3))


def face_normals(vertices: ArrFlt, faces: ArrInt, flat: bool = False) -> ArrFlt:
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :param flat: bool
    :return: [batch size, number of faces, 3, 3]
    """
    np: Any = jnp if isinstance(vertices, jax.Array) else onp

    if faces.ndim == 2:
        faces = faces[None, ...]
    if faces.shape[0] != vertices.shape[0]:
        assert faces.shape[0] == 1
        faces = np.broadcast_to(faces, (vertices.shape[0], faces.shape[0], faces.shape[0]))
    assert vertices.ndim == 3
    assert faces.ndim == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    if not flat:
        v_normals = vertex_normals(vertices, faces)
        return vert2face_attrs(v_normals, faces)
    else:
        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]

        faces = faces + (np.arange(bs, dtype=faces.dtype) * nv)[:, None, None]  # expanded faces
        vertices_faces = np.reshape(vertices, (bs * nv, 3))[faces]
        vertices_faces = np.reshape(vertices_faces, (-1, 3, 3))

        nx = np.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        )
        nx = nx / (np.linalg.norm(nx, axis=1, keepdims=True) + 1e-6)

        nx = np.tile(np.expand_dims(nx, -2), (1, 3, 1))
        nx = np.reshape(nx, (bs, nf, 3, 3))
        return nx
        # return nx.unsqueeze(-2).repeat(1, 3, 1).view(bs, nf, 3, 3)
