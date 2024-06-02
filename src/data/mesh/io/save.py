import numpy as np
import plyfile
import torch


def _to_np(t):
    if t is not None and torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    return t


def save_obj(fname: str, verts, faces, **aux):
    # get aux information
    verts_rgb = aux.get("verts_rgb")
    # move into numpy
    verts = _to_np(verts)
    faces = _to_np(faces)
    verts_rgb = _to_np(verts_rgb)

    with open(fname, "w") as fp:
        for i, vert in enumerate(verts):
            fp.write(f"v {vert[0]} {vert[1]} {vert[2]}")
            if verts_rgb is not None:
                fp.write(f" {verts_rgb[i][0]} {verts_rgb[i][1]} {verts_rgb[i][2]}\n")
            else:
                fp.write("\n")
        for face in faces:
            fp.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_ply(fname: str, verts, faces, **aux):
    verts = _to_np(verts)
    faces = _to_np(faces)
    verts = list(tuple(xyz) for xyz in verts.astype(np.float32))
    faces = list((tuple(xyz),) for xyz in faces.astype(np.int32))

    verts = np.array(verts, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    faces = np.array(faces, dtype=[("vertex_indices", "i4", (3,))])
    plydata = plyfile.PlyData(
        [plyfile.PlyElement.describe(verts, "vertex"), plyfile.PlyElement.describe(faces, "face")]
    )
    plydata.write(fname)
