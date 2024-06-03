import logging
import os
import pickle
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from assets import ASSETS_ROOT

from ..io import denormalize, normalize

logger = logging.getLogger(__name__)


def torch_getAffineTransform(src, tar):
    """
    * Calculates coefficients of affine transformation
    * which maps (xi,yi) to (ui,vi), (i=1,2,3):
    *
    * ui = c00*xi + c01*yi + c02
    *
    * vi = c10*xi + c11*yi + c12
    *
    * Coefficients are calculated by solving linear system:
    * / x0 y0  1  0  0  0 \ /c00\ /u0\
    * | x1 y1  1  0  0  0 | |c01| |u1|
    * | x2 y2  1  0  0  0 | |c02| |u2|
    * |  0  0  0 x0 y0  1 | |c10| |v0|
    * |  0  0  0 x1 y1  1 | |c11| |v1|
    * \  0  0  0 x2 y2  1 / |c12| |v2|
    *
    * where:
    *   cij - matrix coefficients
    """

    assert tar.size(0) == src.size(0)
    bsz = src.size(0)

    tar = tar.permute(0, 2, 1).contiguous().view(bsz, 6, 1)
    src = F.pad(src, (0, 1), mode="constant", value=1)
    zeros = torch.zeros_like(src)
    A = torch.cat(
        (
            torch.cat((src, zeros), dim=-1),
            torch.cat((zeros, src), dim=-1),
        ),
        dim=-2,
    )

    x = torch.solve(tar, A)[0]
    x = x.view(bsz, 2, 3)
    assert not x.isnan().any()
    return x


def cython_generate_mask(pts, hw, triangle_indices):
    try:
        from . import mask_utils
    except ImportError:
        os.system(f"cd {os.path.dirname(__file__)} && python3 setup.py build_ext --inplace")
        from . import mask_utils

    # generate triangle masks
    tri_masks, face_mask = mask_utils.generate_masks(pts, triangle_indices, *hw)
    # generate bbox
    bbox = []
    for tri in triangle_indices:
        tri_pts = np.int32([pts[tri[0]], pts[tri[1]], pts[tri[2]]])
        minx, maxx = min(tri_pts[:, 0]), max(tri_pts[:, 0])
        miny, maxy = min(tri_pts[:, 1]), max(tri_pts[:, 1])
        bbox.append(((minx, miny), (maxx, maxy)))
    tri_masks = np.asarray(tri_masks, dtype=np.float32)
    face_mask = np.asarray(face_mask, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.int32)
    return tri_masks, face_mask, bbox


def cv_generate_mask(pts, hw, triangle_indices):
    # generate bbox
    bbox = []
    tri_masks = []
    face_mask = None
    for tri in triangle_indices:
        tri_pts = np.int32([pts[tri[0]], pts[tri[1]], pts[tri[2]]])
        minx, maxx = min(tri_pts[:, 0]), max(tri_pts[:, 0])
        miny, maxy = min(tri_pts[:, 1]), max(tri_pts[:, 1])
        bbox.append(((minx, miny), (maxx, maxy)))
        mask = np.zeros((int(hw[0]), int(hw[1]), 2), np.float32)
        cv2.fillConvexPoly(mask, tri_pts, (1, 1, 1))
        tri_masks.append(mask)
        if face_mask is None:
            face_mask = mask
        else:
            face_mask = np.logical_or(face_mask, mask)
    tri_masks = np.asarray(tri_masks, dtype=np.float32)
    face_mask = np.asarray(face_mask, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.int32)
    return tri_masks, face_mask, bbox


class FaceWarpingLayer(torch.nn.Module):
    def __init__(
        self,
        template_points: Union[np.ndarray, torch.Tensor],
        triangle_indices: Iterable[Tuple[int, int, int]],
        target_resolution: Tuple[int, int] = (256, 256),
        cached_masks_name: Optional[str] = None,
    ):
        super().__init__()
        assert template_points.ndim == 2 and template_points.shape[1] == 2, "'template_points' should be (N_Points, 2)"
        assert template_points.dtype in [np.float32, torch.float32], "'template_points' should be np or torch float32"
        assert -1 <= template_points.min() and template_points.max() <= 1
        if not torch.is_tensor(template_points):
            template_points = torch.tensor(template_points, dtype=torch.float32)

        self._triangle_indices = np.asarray(triangle_indices)
        self.register_buffer("_tmpl_pts", template_points[None, ...])
        self._tmpl_hw = (target_resolution[1], target_resolution[0])
        self._target_resolution = target_resolution

        # template masks
        cached_masks_path: Optional[str] = None
        if cached_masks_name is not None:
            cached_masks_path = os.path.join(
                ASSETS_ROOT,
                ".face_warping_mask",
                f"{cached_masks_name}_{'x'.join(str(x) for x in target_resolution)}.pkg",
            )
        face_mask, tri_masks, _ = self._generate_mask(template_points.numpy(), self._tmpl_hw, cached_masks_path)
        self.register_buffer("_tmpl_face_mask", torch.tensor(face_mask)[None, ...])
        self.register_buffer("_tmpl_tri_masks", torch.tensor(tri_masks)[None, ...])
        self.register_buffer("_bool_face_mask", torch.tensor(face_mask[:, :, 0] == 1.0)[None, None, ...])
        # get valid indices
        self._pixel_indices = []
        pixel_mask = face_mask[..., 0]
        for i, x in enumerate(pixel_mask.flatten(order="C")):
            if x == 1:
                self._pixel_indices.append(i)
        assert pixel_mask.sum() == len(self._pixel_indices)

    @property
    def pixel_indices(self):
        return self._pixel_indices

    @property
    def resolution(self):
        return self._target_resolution

    def _generate_mask(self, pts, hw, cache_path):
        if cache_path is not None and os.path.exists(cache_path):
            logger.info(f"Load masks for triangles from {cache_path}")
            with open(cache_path, "rb") as fp:
                _cached = pickle.load(fp)
                bbox_list = _cached["bbox_list"]
                face_mask = _cached["face_mask"]
                tri_masks = _cached["tri_masks"]
                assert face_mask.max() == 1.0
                assert face_mask.min() == 0.0
        else:
            # generate masks
            logger.info("Genrating masks for triangles...")
            pts = denormalize(pts, (hw[1], hw[0]))
            tri_masks, face_mask, bbox_list = cython_generate_mask(pts, hw[:2], self._triangle_indices)
            bbox_list = np.asarray([], dtype=np.int32)

            # dump pickle
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as fp:
                    pickle.dump(
                        dict(
                            tri_masks=tri_masks,
                            face_mask=face_mask,
                            bbox_list=bbox_list,
                        ),
                        fp,
                    )
        # # debug
        # print(face_mask.shape, tri_masks.shape)
        # cv2.imshow('face_mask', face_mask[..., [0, 0, 0]])
        # cv2.waitKey()
        # for tri_mask in tri_masks:
        #     cv2.imshow('face_mask', tri_mask[..., [0, 0, 0]])
        #     cv2.waitKey()
        return face_mask, tri_masks, bbox_list

    def _get_mask(self, bsz, target_pts, target_hw):
        if target_pts is None:
            return (
                self._tmpl_face_mask.expand(bsz, -1, -1, -1),
                self._tmpl_tri_masks.expand(bsz, -1, -1, -1, -1),
                None,
            )
        else:
            assert bsz == target_pts.size(0)
            target_pts = denormalize(target_pts, (float(target_hw[1]), float(target_hw[0])))
            batch_face_mask = []
            batch_tri_masks = []
            for lmks in target_pts:
                face_mask, tri_masks, _ = self._generate_mask(lmks.detach().cpu().numpy(), target_hw, None)
                batch_face_mask.append(face_mask)
                batch_tri_masks.append(tri_masks)
            batch_face_mask = np.asarray(batch_face_mask, np.float32)
            batch_tri_masks = np.asarray(batch_tri_masks, np.float32)
            return (
                torch.FloatTensor(batch_face_mask).to(target_pts.device),
                torch.FloatTensor(batch_tri_masks).to(target_pts.device),
                None,
            )

    def _get_inv_affine_transform(self, i_tri, pts, target_pts):
        if target_pts is None:
            tar = self._tmpl_pts.expand_as(pts[:, : self._tmpl_pts.shape[1]])
        else:
            tar = target_pts
        assert pts.shape[0] == tar.shape[0], "batch size mismatch! src {}, tar {}".format(pts.shape[0], tar.shape[0])
        tri = self._triangle_indices[i_tri]
        src = torch.stack((pts[:, tri[0]], pts[:, tri[1]], pts[:, tri[2]]), dim=1)
        tar = torch.stack((tar[:, tri[0]], tar[:, tri[1]], tar[:, tri[2]]), dim=1)
        M = torch_getAffineTransform(tar, src)
        return M

    def forward(self, *inputs, **kwargs):
        return self.warp_face_image(*inputs, **kwargs)

    def warp_face_image(self, img, pts):
        # assert -1 <= pts.min() and pts.max() <= 1

        squeeze = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            pts = pts.unsqueeze(0)
            squeeze = True

        # get batch size
        bsz = img.shape[0]
        if img.shape[1] == 4:
            img = img[:, :3]
        assert img.shape[1] == 3, "wrong img shape: {}".format(img.shape)

        # get masks
        face_mask = self._tmpl_face_mask
        tri_masks = self._tmpl_tri_masks

        # fill affine_grid from each triangle
        grid = torch.zeros((bsz, *self._tmpl_hw, 2), dtype=torch.float32, requires_grad=True).to(img.device)
        for i_tri in range(len(self._triangle_indices)):
            _inv_trans = self._get_inv_affine_transform(i_tri, pts, self._tmpl_pts.expand(pts.shape[0], -1, -1))
            _grid_size = (bsz, img.shape[1], *self._tmpl_hw)
            grid_i = F.affine_grid(_inv_trans, _grid_size, align_corners=False)
            grid_i = grid_i * tri_masks[:, i_tri]
            grid = grid + grid_i

        # warp image
        warp_grid = grid * face_mask + torch.full_like(grid, 2) * (1 - face_mask)  # fill with out of range src
        outputs = F.grid_sample(img, warp_grid, align_corners=False)
        outputs = torch.where(self._bool_face_mask, outputs, torch.zeros_like(outputs))

        if squeeze:
            outputs = outputs.squeeze(0)
        return outputs


def cv_warp(src_img, src_pts, dst_pts, dst_res, triangle_indices):
    assert src_img.ndim == 3 and src_img.shape[-1] == 3
    assert src_pts.ndim == 2
    assert dst_pts.ndim == 2

    triangle_indices = np.asarray(triangle_indices)

    src_pts = denormalize(src_pts, (src_img.shape[0], src_img.shape[1]))
    dst_pts = denormalize(dst_pts, (dst_res[1], dst_res[0])).astype(np.int32)

    canvas = np.zeros((dst_res[1], dst_res[0], 3), dtype=src_img.dtype)
    tri_masks, _, _ = cv_generate_mask(dst_pts, (dst_res[1], dst_res[0]), triangle_indices)
    for tri_idx, tri_mask in zip(triangle_indices, tri_masks):
        src_tri = np.asarray([src_pts[tri_idx[i]] for i in range(3)])
        dst_tri = np.asarray([dst_pts[tri_idx[i]] for i in range(3)])
        M = cv2.getAffineTransform(src_tri, dst_tri.astype(np.float32))
        dst_part = cv2.warpAffine(src_img, M, dst_res)
        canvas = canvas * (1.0 - tri_mask[:, :, 0:1]) + dst_part * tri_mask[:, :, 0:1]
    return canvas
