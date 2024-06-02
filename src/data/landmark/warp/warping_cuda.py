import os

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import torch

x = torch.cuda.FloatTensor(8)  # important: share context with torch.cuda


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer():
        return self.t.data_ptr()


class WarpingCuda:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        self._mod = self._module_from_cu(os.path.join(dirname, "warping_cuda.cu"))
        self._warp_fn = self._mod.get_function("warp_image")
        self._mask_fn = self._mod.get_function("face_mask")

    def _module_from_cu(self, path):
        from pycuda.compiler import SourceModule

        with open(path) as fin:
            mod = SourceModule(fin.read())
        return mod

    def _get_trimesh(self, pts, triangles=None, dtype=np.float32):
        if triangles is None:
            triangles = lmksutils.face_triangles
        trimesh = np.zeros((len(triangles), 6), dtype)
        for i, tri in enumerate(triangles):
            trimesh[i][0:2] = pts[tri[0]]
            trimesh[i][2:4] = pts[tri[1]]
            trimesh[i][4:6] = pts[tri[2]]
        return trimesh.astype(dtype)

    def _get_affine_transforms(self, src_trimesh, tar_trimesh):
        assert len(src_trimesh) == len(tar_trimesh)
        tforms = np.zeros((len(src_trimesh), 6), np.float32)
        for i in range(len(src_trimesh)):
            cv_res = cv2.getAffineTransform(np.reshape(src_trimesh[i], (3, 2)), np.reshape(tar_trimesh[i], (3, 2)))
            tforms[i] = cv_res.flatten()
        return tforms

    def generate_face_mask(self, image_shape, pts, pts_normalized, triangles=None):
        if pts_normalized:
            pts = lmksutils.denormalize_points(pts, *image_shape[:2])
        if triangles is None:
            triangles = lmksutils.face_triangles
        masks = np.zeros((len(triangles), *image_shape[:2], 1), np.float32)
        full_mask = np.zeros((*image_shape[:2], 1), np.float32)
        trimesh = self._get_trimesh(pts, triangles=triangles)

        block = (32, 32, 1)
        grid = (int(np.ceil(image_shape[1] / block[0])), int(np.ceil(image_shape[0] / block[1])))
        self._mask_fn(
            drv.Out(masks),
            drv.Out(full_mask),
            drv.In(trimesh),
            drv.In(np.asarray(masks.shape, np.int32)),
            block=block,
            grid=grid,
        )
        return masks.astype(np.float32), full_mask.astype(np.float32)

    def warp_face_image(self, out, src, out_pts, src_pts, pts_normalized):
        """warp face from src to out"""

        def _isch(shape, i):
            return shape[i] in [1, 2, 3, 4]

        assert isinstance(out, np.ndarray)
        assert isinstance(src, np.ndarray)
        status = dict(img_format="HWC")
        if _isch(out.shape, 2) and _isch(src.shape, 2):
            status["img_format"] = "HWC"
        elif _isch(out.shape, 0) and _isch(src.shape, 0):
            status["img_format"] = "CHW"
            out = np.transpose(out, (1, 2, 0))
            src = np.transpose(src, (1, 2, 0))
        else:
            raise ValueError("out's shape {}, src's shape {}.".format(out.shape, src.shape))

        # denormalize
        if pts_normalized:
            out_pts = lmksutils.denormalize_points(out_pts, out.shape[0], out.shape[1])
            src_pts = lmksutils.denormalize_points(src_pts, src.shape[0], src.shape[1])
        # get trimesh
        out_trimesh = self._get_trimesh(out_pts)
        src_trimesh = self._get_trimesh(src_pts)
        # make sure the dtype is float32
        out = out.astype(np.float32)
        src = src.astype(np.float32)

        tforms = self._get_affine_transforms(out_trimesh, src_trimesh)
        block = (32, 32, 1)
        grid = (int(np.ceil(out.shape[1] / 32)), int(np.ceil(out.shape[0] / 32)))
        self._warp_fn(
            drv.Out(out),
            drv.In(src),
            drv.In(out_trimesh),
            drv.In(tforms),
            drv.In(np.asarray(out.shape, np.int32)),
            drv.In(np.asarray(src.shape, np.int32)),
            drv.In(np.asarray(out_trimesh.shape, np.int32)),
            block=block,
            grid=grid,
        )

        if status["img_format"] == "CHW":
            out = np.transpose(out, (2, 0, 1))
        return out


warping = WarpingCuda()
