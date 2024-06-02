import os
from functools import lru_cache

from .assets import ASSETS_ROOT

DATA_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(DATA_ROOT, "datasets")

_vocaset_template_verts_dict = dict()
_vocaset_template_tris = None


def get_vocaset_template_vertices(speaker="TMPL"):
    global _vocaset_template_verts_dict
    global _vocaset_tmplate_tris

    if _vocaset_template_verts_dict.get(speaker) is None:
        from src.data.mesh.io import load_mesh

        verts, tris, _ = load_mesh(os.path.join(DATA_ROOT, "vocaset_templates", f"{speaker}.ply"))
        _vocaset_template_verts_dict[speaker] = verts
        if _vocaset_template_tris is None:
            _vocaset_template_tris = tris

    return _vocaset_template_verts_dict[speaker]


def get_vocaset_template_triangles():
    global _vocaset_template_tris

    if _vocaset_template_tris is None:
        from src.data.mesh.io import load_mesh

        _, _vocaset_template_tris, _ = load_mesh(
            os.path.join(DATA_ROOT, "vocaset_templates", "FaceTalk_170725_00137_TA.ply")
        )

    return _vocaset_template_tris


@lru_cache(maxsize=5)
def get_selection_triangles(part):
    from src.data.mesh.io import load_mesh

    _, tris, _ = load_mesh(os.path.join(DATA_ROOT, "assets", "selection", f"{part}.obj"))
    return tris


@lru_cache(maxsize=5)
def get_selection_vidx(part):
    with open(get_selection_vidx_path(part)) as fp:
        line = " ".join([x.strip() for x in fp])
        vidx = [int(x) for x in line.split()]
    return vidx


def get_selection_vidx_path(part):
    return os.path.join(DATA_ROOT, "assets", "selection", f"{part}.txt")


def get_selection_obj(part):
    return os.path.join(DATA_ROOT, "assets", "selection", f"{part}.obj")
