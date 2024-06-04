import os
from functools import lru_cache

ASSETS_ROOT = os.path.abspath(os.path.dirname(__file__))
PATH_DEFAULT_FONT = os.path.join(ASSETS_ROOT, "fonts", "Roboto-Regular.ttf")
PATH_COLORBAR_JET = os.path.join(ASSETS_ROOT, "MATLAB_jet.png")

with open(os.path.join(ASSETS_ROOT, "metric_face_vidx.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    METRIC_FACE_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "metric_lips_vidx.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    LIPS_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "metric_lower_vidx.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    METRIC_LOWER_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "eyes_vidx.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    EYES_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "face_lower.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    FACE_LOWER_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "eyebrows.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    EYEBROWS_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "face.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    FACE_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "face_neye.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    FACE_NEYE_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "face_noeyeballs.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    FACE_NOEYEBALLS_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "face_below_eye.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    FACE_BELOW_EYE_VIDX = [int(x) for x in line.split()]
with open(os.path.join(ASSETS_ROOT, "selection", "eyes_above.txt")) as fp:
    line = " ".join(x.strip() for x in fp)
    EYES_ABOVE_VIDX = [int(x) for x in line.split()]

INN_LIP_VIDX = [
    3533,
    2785,
    2784,
    2855,
    2863,
    2836,
    2835,
    2839,
    2843,
    2717,
    2716,
    2889,
    2884,
    2885,
    2929,
    2934,
    3513,
    1836,
    1827,
    1778,
    1777,
    1782,
    1580,
    1581,
    1728,
    1722,
    1718,
    1719,
    1748,
    1740,
    1667,
    1668,
]
INN_LIP_UPPER_VIDX = [2839, 2835, 2836, 2863, 2855, 2784, 2785, 3533, 1668, 1667, 1740, 1748, 1719, 1718, 1722]
INN_LIP_LOWER_VIDX = [2717, 2716, 2889, 2884, 2885, 2929, 2934, 3513, 1836, 1827, 1778, 1777, 1782, 1580, 1581]

__all__ = [
    "ASSETS_ROOT",
    "PATH_DEFAULT_FONT",
    "PATH_COLORBAR_JET",
    "METRIC_FACE_VIDX",
    "LIPS_VIDX",
    "METRIC_LOWER_VIDX",
    "FACE_NEYE_VIDX",
    "FACE_NOEYEBALLS_VIDX",
    "EYES_VIDX",
    "FACE_LOWER_VIDX",
    "EYEBROWS_VIDX",
    "EYES_ABOVE_VIDX",
    "FACE_BELOW_EYE_VIDX",
    "FACE_VIDX",
    "INN_LIP_VIDX",
    "INN_LIP_UPPER_VIDX",
    "INN_LIP_LOWER_VIDX",
]


DATA_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(DATA_ROOT, "datasets")

_vocaset_template_verts_dict = dict()
_vocaset_template_tris = None


def get_vocaset_template_vertices(speaker="TMPL"):
    global _vocaset_template_verts_dict
    global _vocaset_template_tris

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
    return os.path.join(DATA_ROOT, "selection", f"{part}.txt")


def get_selection_obj(part):
    return os.path.join(DATA_ROOT, "selection", f"{part}.obj")
