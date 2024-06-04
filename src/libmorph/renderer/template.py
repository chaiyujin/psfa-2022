import os

import torch

from ..utils.attribute import vert2face_attrs
from ..utils.tensor_props import TensorProperties


class MeshTemplate(TensorProperties):
    def __init__(self, obj_filename):
        obj_filename = os.path.expanduser(obj_filename)
        assert os.path.exists(obj_filename), "Failed to find {}".format(obj_filename)
        # load the template obj file
        verts, verts_uvs, faces_verts_idx, faces_textures_idx = _load_obj(obj_filename)

        # * scale uv into -1~1 for grid_sample
        if verts_uvs.nelement() == 0:
            uvcoords = None
            NO_uvcoords = None
            NO_tri_uvcoords = None
        else:
            uvcoords = verts_uvs[None, ...]
            NO_uvcoords = uvcoords.clone() * 2.0 - 1.0
            NO_uvcoords[..., 1] *= -1.0
            NO_tri_uvcoords = vert2face_attrs(NO_uvcoords, faces_textures_idx[None, ...])

        super().__init__(
            vertices=verts[None, ...],
            triangles=faces_verts_idx[None, ...].int(),
            uv_triangles=faces_textures_idx[None, ...].int(),
            uvcoords=uvcoords,
            NO_uvcoords=NO_uvcoords,
            NO_tri_uvcoords=NO_tri_uvcoords,
            # shared by batch, not computed as batch
            __shared_keys__=("L",),
        )


def _load_obj(obj_filename):
    with open(obj_filename) as fp:
        (
            verts,
            normals,
            verts_uvs,
            faces_verts_idx,
            faces_normals_idx,
            faces_textures_idx,
            faces_materials_idx,
            material_names,
            mtl_path,
        ) = _parse_obj(fp, os.path.dirname(obj_filename))
    return (
        torch.tensor(verts, dtype=torch.float32),
        torch.tensor(verts_uvs, dtype=torch.float32),
        torch.tensor(faces_verts_idx, dtype=torch.int) - 1,
        torch.tensor(faces_textures_idx, dtype=torch.int) - 1,
    )


def _parse_obj(f, data_dir: str):
    """
    Load a mesh from a file-like object. See load_obj function for more details
    about the return values.
    """
    verts, normals, verts_uvs = [], [], []
    faces_verts_idx, faces_normals_idx, faces_textures_idx = [], [], []
    faces_materials_idx = []
    material_names = []
    mtl_path = None

    lines = [line.strip() for line in f]

    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    materials_idx = -1

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("mtllib"):
            if len(tokens) < 2:
                raise ValueError("material file name is not specified")
            # NOTE: only allow one .mtl file per .obj.
            # Definitions for multiple materials can be included
            # in this one .mtl file.
            mtl_path = line[len(tokens[0]) :].strip()  # Take the remainder of the line
            mtl_path = os.path.join(data_dir, mtl_path)
        elif len(tokens) and tokens[0] == "usemtl":
            material_name = tokens[1]
            # materials are often repeated for different parts
            # of a mesh.
            if material_name not in material_names:
                material_names.append(material_name)
                materials_idx = len(material_names) - 1
            else:
                materials_idx = material_names.index(material_name)
        elif line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError("Texture %s does not have 2 values. Line: %s" % (str(tx), str(line)))
            verts_uvs.append(tx)
        elif line.startswith("vn "):  # Line is a normal.
            norm = [float(x) for x in tokens[1:4]]
            if len(norm) != 3:
                msg = "Normal %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(norm), str(line)))
            normals.append(norm)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            _parse_face(
                line,
                tokens,
                materials_idx,
                faces_verts_idx,
                faces_normals_idx,
                faces_textures_idx,
                faces_materials_idx,
            )

    return (
        verts,
        normals,
        verts_uvs,
        faces_verts_idx,
        faces_normals_idx,
        faces_textures_idx,
        faces_materials_idx,
        material_names,
        mtl_path,
    )


def _parse_face(
    line,
    tokens,
    material_idx,
    faces_verts_idx,
    faces_normals_idx,
    faces_textures_idx,
    faces_materials_idx,
):
    face = tokens[1:]
    face_list = [f.split("/") for f in face]
    face_verts = []
    face_normals = []
    face_textures = []

    for vert_props in face_list:
        # Vertex index.
        face_verts.append(int(vert_props[0]))
        if len(vert_props) > 1:
            if vert_props[1] != "":
                # Texture index is present e.g. f 4/1/1.
                face_textures.append(int(vert_props[1]))
            if len(vert_props) > 2:
                # Normal index present e.g. 4/1/1 or 4//1.
                face_normals.append(int(vert_props[2]))
            if len(vert_props) > 3:
                raise ValueError(
                    "Face vertices can ony have 3 properties. \
                                Face vert %s, Line: %s"
                    % (str(vert_props), str(line))
                )

    # Triplets must be consistent for all vertices in a face e.g.
    # legal statement: f 4/1/1 3/2/1 2/1/1.
    # illegal statement: f 4/1/1 3//1 2//1.
    # If the face does not have normals or textures indices
    # fill with pad value = -1. This will ensure that
    # all the face index tensors will have F values where
    # F is the number of faces.
    if len(face_normals) > 0:
        if not (len(face_verts) == len(face_normals)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )
    else:
        face_normals = [-1] * len(face_verts)  # Fill with -1
    if len(face_textures) > 0:
        if not (len(face_verts) == len(face_textures)):
            raise ValueError(
                "Face %s is an illegal statement. \
                        Vertex properties are inconsistent. Line: %s"
                % (str(face), str(line))
            )
    else:
        face_textures = [-1] * len(face_verts)  # Fill with -1

    # Subdivide faces with more than 3 vertices.
    # See comments of the load_obj function for more details.
    for i in range(len(face_verts) - 2):
        faces_verts_idx.append((face_verts[0], face_verts[i + 1], face_verts[i + 2]))
        faces_normals_idx.append((face_normals[0], face_normals[i + 1], face_normals[i + 2]))
        faces_textures_idx.append((face_textures[0], face_textures[i + 1], face_textures[i + 2]))
        faces_materials_idx.append(material_idx)
