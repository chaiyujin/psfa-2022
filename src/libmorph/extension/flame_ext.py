import json
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..morphable.flame import lbs
from ..utils import Struct, to_np, to_tensor

try:
    from .contour import contour_finder
except ImportError:
    import sys

    _dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"cd {os.path.join(_dir, 'contour')} && {sys.executable} setup.py build_ext --inplace"
    os.system(cmd)
    from .contour import contour_finder


def _load_index(fpath):
    with open(fpath) as fp:
        line = " ".join(x.strip() for x in fp)
    return sorted([int(x) for x in line.split()])


class FLAMEExt(nn.Module):
    def __init__(self, config):
        super().__init__()

        with open(config.flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.register_buffer("faces_tensor", to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        # * Masks
        self._init_masks(config)

        # * FW75 landmarks
        self._init_fw75(config)

        # * i-Bug68 landmarks
        self._init_ibug68(config)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                     Masks                                                    * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _init_masks(self, config):
        mask_path = config.mask_path
        assert os.path.exists(mask_path), "Failed to find {}".format(mask_path)
        with open(mask_path, "rb") as fp:
            self._masks_vidx = pickle.load(fp, encoding="latin1")

        uv_masks_dir = config.uv_masks_dir
        self._valid_uv_masks = []
        for name in ["mask_eye", "mask_face", "mask_face_weye"]:
            uv_mask_path = os.path.join(uv_masks_dir, name + ".png")
            assert os.path.exists(uv_mask_path), "Failed to find {}".format(uv_mask_path)
            uv_mask = np.clip(cv2.imread(uv_mask_path)[..., 0] / 255.0, 0, 1)[None, None]
            self.register_buffer(f"_uv_{name}", torch.tensor(uv_mask, dtype=torch.float32))
            self._valid_uv_masks.append(name[5:])

    def uv_mask(self, name, batch_size=1) -> torch.Tensor:
        attr_name = f"_uv_mask_{name}"
        if name not in self._valid_uv_masks:
            raise ValueError("No uv mask for '{}', only {}".format(name, self._valid_uv_masks))
        ret = getattr(self, attr_name)
        return ret.expand(batch_size, -1, -1, -1) if batch_size > 1 else ret

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                          FacewareHouse 75 landmarks                                          * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _init_fw75(self, config):
        # * for fw75
        embed_path = config.fw75_embed_path
        with open(embed_path) as fp:
            data = json.load(fp)
        faces_idx = []
        bary_coords = []
        for d in data:
            faces_idx.append(d[0])
            bary_coords.append((d[1], d[2], 1.0 - d[1] - d[2]))

        self.register_buffer("_fw75_all_faces_idx", torch.tensor(faces_idx, dtype=torch.long)[None, ...])
        self.register_buffer("_fw75_all_barycoord", torch.tensor(bary_coords, dtype=torch.float32)[None, ...])

        # * for face contour finding
        cntr_fidx = _load_index(config.contour_mask_path)
        face_cntr_fidx = _load_index(config.face_contour_mask_path)
        self._cntr_fidx_flags = np.zeros((config.contour_target_n_triangles,), dtype=np.uint8)
        self._cntr_fidx_flags[cntr_fidx] = 1
        self._face_cntr_fidx_flags = np.zeros((config.contour_target_n_triangles,), dtype=np.uint8)
        self._face_cntr_fidx_flags[face_cntr_fidx] = 1

    def get_static_fw75_landmarks(self, vertices):
        N = vertices.shape[0]
        return lbs.vertices2landmarks(
            vertices, self.faces_tensor, self._fw75_all_faces_idx.repeat(N, 1), self._fw75_all_barycoord.repeat(N, 1, 1)
        )

    def find_contour(self, vertices, rast_out):
        rast_cpu = rast_out.detach().cpu().numpy()
        faces_idx_list, barycoord_list, mask = [], [], []
        max_points = 0
        for bi in range(rast_cpu.shape[0]):
            faces_idx, barycoord, marking = contour_finder.find_contour(
                rast_cpu[bi], self._cntr_fidx_flags, self._face_cntr_fidx_flags
            )
            if len(faces_idx) > 0:
                mask.append(1)
                faces_idx = np.asarray(faces_idx, dtype=np.int64)
                barycoord = np.asarray(barycoord, dtype=np.float32)
                max_points = max(max_points, len(faces_idx))
                faces_idx_list.append(faces_idx)
                barycoord_list.append(barycoord)
            else:
                mask.append(0)
                faces_idx_list.append(np.zeros((1,), dtype=np.int64))
                barycoord_list.append(np.zeros((1, 3), dtype=np.float32))
            # if bi == 0:
            #     cv2.imshow('marking', marking)
            #     cv2.waitKey(1)

        # padding
        faces_idx_list = [np.pad(x, [[0, max_points - len(x)]], "edge") for x in faces_idx_list]
        barycoord_list = [np.pad(x, [[0, max_points - len(x)], [0, 0]], "edge") for x in barycoord_list]

        # query points
        cntr_faces_idx = torch.tensor(faces_idx_list, dtype=torch.long, device=vertices.device)
        cntr_barycoord = torch.tensor(barycoord_list, dtype=torch.float32, device=vertices.device)
        cntr_mask = torch.tensor(mask, dtype=torch.bool, device=vertices.device)
        points = lbs.vertices2landmarks(vertices, self.faces_tensor, cntr_faces_idx, cntr_barycoord)
        # print(points.shape, cntr_mask.shape)
        return points, cntr_mask

    @staticmethod
    def correct_fw75_contour(lmks_pred, cntr_pts, cntr_mask, lmks_real):
        def _find_closest(pts, qry):
            N, L = pts.shape[0], None
            if pts.ndim == 4:
                L = pts.shape[1]
                pts = pts.view(N * L, *pts.shape[2:])
                qry = qry.view(N * L, *qry.shape[2:])

            assert pts.ndim == 3 and qry.ndim == 3

            bsz = qry.shape[0]
            dist = torch.norm((qry[..., :2].unsqueeze(-2) - pts[..., :2].unsqueeze(-3)), dim=-1).detach()
            cidx = torch.argmin(dist, dim=-1)
            didx = torch.arange(bsz, device=cidx.device) * pts.shape[-2]
            index = (cidx + didx[:, None]).view(bsz * qry.shape[-2])
            val_flatten = torch.index_select(pts.view(-1, pts.shape[-1]), dim=0, index=index)
            val = val_flatten.view(bsz, qry.shape[-2], val_flatten.shape[-1])

            if L is not None:
                val = val.view(N, L, *val.shape[1:])
            return val

        cntr_mask = cntr_mask[..., None, None]
        # query contour points for fw75
        cntr_pred = _find_closest(cntr_pts, lmks_real[..., :15, :])
        # mask out invalid cntr with predicted static landmarks
        cntr_pred = torch.where(cntr_mask, cntr_pred, lmks_pred[..., :15, :])
        # replace contour points with found ones
        return torch.cat((cntr_pred, lmks_pred[..., 15:, :]), dim=-2)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                               Ibug68 Landmarks                                               * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _init_ibug68(self, config):
        # fmt: off
        lmk_embeddings = np.load(config.ibug68_embed_path, allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer("_ibug68_all_faces_idx", torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long())
        self.register_buffer("_ibug68_all_barycoord", torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(torch.float32))
        self.register_buffer("_ibug68_inn_faces_idx", torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long())
        self.register_buffer("_ibug68_inn_barycoord", torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(torch.float32))
        self.register_buffer("_ibug68_dyn_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long())
        self.register_buffer("_ibug68_dyn_barycoord", lmk_embeddings["dynamic_lmk_bary_coords"].to(torch.float32))
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))
        # fmt: on

    def get_static_ibug68_landmarks(self, vertices):
        landmarks3d = lbs.vertices2landmarks(
            vertices,
            self.faces_tensor,
            self._ibug68_all_faces_idx.repeat(vertices.shape[0], 1),
            self._ibug68_all_barycoord.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def _check_params(self, params, batch_size, n_params):
        if params is not None:
            assert params.ndim == 2 and params.shape[1] == n_params
            if params.shape[0] != batch_size:
                assert params.shape[0] == 1
                params = params.expand(batch_size, -1)
            return params
        # default value on None
        val = torch.zeros((1, n_params), dtype=torch.float32, device=self._ibug68_all_faces_idx.device)
        return val.expand(batch_size, -1)

    def get_dynamic_ibug68_landmarks(
        self, vertices, rotat=None, neck_pose=None, jaw_pose=None, eye_pose=None, **kwargs
    ):
        # fmt: off
        N = vertices.shape[0]
        rotat     = self._check_params(rotat,     N, 3)
        neck_pose = self._check_params(neck_pose, N, 3)
        jaw_pose  = self._check_params(jaw_pose,  N, 3)
        eye_pose  = self._check_params(eye_pose,  N, 6)
        full_pose = torch.cat([rotat, neck_pose, jaw_pose, eye_pose], dim=1)
        # fmt: on

        # inner face
        inn_faces_idx = self._ibug68_inn_faces_idx.unsqueeze(dim=0).expand(N, -1)
        inn_barycoord = self._ibug68_inn_barycoord.unsqueeze(dim=0).expand(N, -1, -1)
        # dynamic contour
        dyn_faces_idx, dyn_barycoord = lbs.find_dynamic_lmk_idx_and_bcoords(
            vertices,
            full_pose,
            self._ibug68_dyn_faces_idx,
            self._ibug68_dyn_barycoord,
            self.neck_kin_chain,
            dtype=vertices.dtype,
        )
        # concat
        lmk_faces_idx = torch.cat([dyn_faces_idx, inn_faces_idx], 1)
        lmk_barycoord = torch.cat([dyn_barycoord, inn_barycoord], 1)

        return lbs.vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_barycoord)
