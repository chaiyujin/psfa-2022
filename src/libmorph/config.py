import os
from dataclasses import dataclass

from assets import ASSETS_ROOT

DATA_DIR = os.path.join(ASSETS_ROOT, "flame-data")


@dataclass
class FLAMEConfig:
    # * For FLAME
    model_path: str = os.path.join(DATA_DIR, "FLAME2020", "generic_model-np.pkl")
    n_shape: int = 100
    n_exp: int = 50

    # * For FLAMETex
    tex_type: str = "BFM"
    tex_path: str = os.path.join(DATA_DIR, "FLAME_Texture", "FLAME_albedo_from_BFM.npz")
    n_tex: int = 50

    # * For FLAMEUtil
    # For i-Bug 68 landmarks
    ibug68_embed_path: str = os.path.join(DATA_DIR, "landmark_embedding.npy")
    # For FacewareHouse 75 landmarks
    fw75_embed_path: str = os.path.join(DATA_DIR, "fw75", "landmarks_fw75.txt")
    # For Contour finding
    contour_target_n_triangles: int = 9976
    contour_mask_path: str = os.path.join(DATA_DIR, "contour_index", "contour_candidate_fidx.txt")
    face_contour_mask_path: str = os.path.join(DATA_DIR, "contour_index", "face_contour_candidate_fidx.txt")
    # For vertices masks
    mask_path: str = os.path.join(DATA_DIR, "FLAME_masks", "FLAME_masks.pkl")
    # uv masks
    uv_masks_dir: str = os.path.join(DATA_DIR, "uv_masks")


@dataclass
class ExtensionConfig:
    # * For i-Bug 68 landmarks
    ibug68_embed_path: str = os.path.join(DATA_DIR, "landmark_embedding.npy")
    flame_model_path: str = os.path.join(DATA_DIR, "FLAME2020", "generic_model-np.pkl")

    # * For FacewareHouse 75 landmarks
    fw75_embed_path: str = os.path.join(DATA_DIR, "fw75", "landmarks_fw75.txt")

    # * For Contour finding
    contour_target_n_triangles: int = 9976
    contour_mask_path: str = os.path.join(DATA_DIR, "contour_index", "contour_candidate_fidx.txt")
    face_contour_mask_path: str = os.path.join(DATA_DIR, "contour_index", "face_contour_candidate_fidx.txt")

    # * For vertices masks
    mask_path: str = os.path.join(DATA_DIR, "FLAME_masks", "FLAME_masks.pkl")

    # * uv masks
    uv_masks_dir: str = os.path.join(DATA_DIR, "uv_masks")
