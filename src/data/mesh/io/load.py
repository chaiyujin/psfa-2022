import os
from typing import Any, Dict, Tuple

import numpy as np
import plyfile


def load_mesh(fname: str, dtype=np.float32, flatten: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(fname)[1]
    if ext == ".obj":
        verts, faces, aux = read_obj_np(fname)
    elif ext == ".ply":
        verts, faces, aux = read_ply_np(fname)
    else:
        raise NotImplementedError(f"Cannot read '{fname}'.")
    verts = verts.astype(dtype)
    faces = faces.astype(np.int32)
    aux["verts_rgb"] = aux["verts_rgb"].astype(dtype)
    if flatten:
        verts = verts.flatten(order="C")
        faces = faces.flatten(order="C")
        aux["verts_rgb"] = aux["verts_rgb"].flatten(order="C")
    return verts, faces, aux


def read_ply_np(ply_path, expect_indices=None):
    plydata = plyfile.PlyData.read(ply_path)
    verts = np.stack((plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]), axis=1).astype(
        np.float32
    )
    faces = np.stack(plydata["face"]["vertex_indices"], axis=0).astype(np.int32)
    # read verts_rgb
    verts_rgb = np.ones_like(verts)
    props = plydata["vertex"].properties
    for prop in props:
        ch = -1
        if prop.name == "red":
            ch = 0
        elif prop.name == "green":
            ch = 1
        elif prop.name == "blue":
            ch = 2
        else:
            continue
        if 0 <= ch <= 2:
            if prop.val_dtype == "u1":
                verts_rgb[:, ch] = plydata["vertex"][prop.name].astype(np.float32) / 255.0
            elif prop.val_dtype in ["f4", "f8"]:
                verts_rgb[:, ch] = plydata["vertex"][prop.name].astype(np.float32)
            else:
                raise NotImplementedError("Unknown ply rgb val_dtype: '{}'".format(prop.val_dtype))
    if expect_indices is not None:
        assert np.all(expect_indices == faces)
    return verts, faces, dict(verts_rgb=verts_rgb)


def read_obj_np(filename_obj, expect_indices=None, normalization=False):
    # load vertices
    with open(filename_obj) as f:
        lines = f.readlines()

    verts = []
    verts_rgb = []
    for line in lines:
        line = line.strip().split()
        if len(line) == 0:
            continue
        if line[0] == "v":
            verts.append([float(v) for v in line[1:4]])
            if 7 <= len(line):
                verts_rgb.append([float(v) for v in line[4:7]])
            else:
                verts_rgb.append([1.0, 1.0, 1.0])
    verts = np.vstack(verts).astype(np.float32)
    verts_rgb = np.vstack(verts_rgb).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
    faces = (np.vstack(faces) - 1).astype(np.int32)

    # normalize into a unit cube centered zero
    if normalization:
        verts -= verts.min(0)[0][None, :]
        verts /= np.abs(verts).max()
        verts *= 2
        verts -= verts.max(0)[0][None, :] / 2

    verts = verts.astype(np.float32)
    faces = faces.astype(np.int32)

    if expect_indices is not None:
        assert np.all(expect_indices == faces)

    return verts, faces, dict(verts_rgb=verts_rgb)
