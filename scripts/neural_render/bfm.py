import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

from src.mm_fitting.libmorph import MeshRasterizer


class BFM(nn.Module):
    def __init__(self, model_path: str = "~/assets/BFM/BFM_model_front.mat"):
        super().__init__()
        model_path = os.path.expanduser(model_path)
        model = loadmat(model_path)

        mean_pos = torch.tensor(model["meanshape"].astype(np.float32).transpose(1, 0))  # mean face shape. [3*N,1]
        _id_base = torch.tensor(model["idBase"].astype(np.float32))  # identity basis. [3*N,80]
        exp_base = torch.tensor(model["exBase"].astype(np.float32))  # expression basis. [3*N,64]
        triangle = torch.tensor(model["tri"])  # vertex indices in each triangle. starts from 1. [F,3]
        # torch.tensor(model["point_buf"])  # triangle indices for each vertex that lies in. starts from 1. [N,8]

        self.register_buffer("mean_pos", mean_pos)
        self.register_buffer("_id_base", _id_base)
        self.register_buffer("exp_base", exp_base)
        self.register_buffer("triangle", triangle)

        # for re-center
        center = mean_pos.view(-1, 3).mean(dim=0, keepdim=True)
        self.register_buffer("center", center.unsqueeze(1))  # [1, 1, 3]

        # init rasterizer
        self.rast = MeshRasterizer(224, os.path.join(os.path.dirname(model_path), "BFM_FRONT_template_uvs.obj"))
        # two kinds of uv coords (-1~1)
        uv_coords = np.load(os.path.join(os.path.dirname(model_path), "uv.npy")) * 2.0 - 1.0  # type: ignore
        self.register_buffer("NO_uvcoords_official", torch.tensor(uv_coords[np.newaxis, :, :2], dtype=torch.float32))
        self.register_buffer("NO_uvcoords_ours", self.rast.template.NO_uvcoords.clone())

        # camera projection: different implementation may use different znear/zfar, but others are same
        self.register_buffer("proj_tf", self._init_camera(znear=0.01, zfar=50))
        self.register_buffer("proj_torch", self._init_camera(znear=5.0, zfar=15.0))

    def _init_camera(self, center=112.0, focal=1015.0, camera_d=10.0, znear=0.01, zfar=50):
        # projection matrix
        def ndc_projection(x=0.1, n=1.0, f=50.0):
            return np.array(
                [[n / x, 0, 0, 0], [0, n / -x, 0, 0], [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)], [0, 0, -1, 0]]
            ).astype(np.float32)

        fov = 2 * np.arctan(center / focal) * 180 / np.pi  # type: ignore
        x = np.tan(np.deg2rad(fov * 0.5)) * znear  # type: ignore
        ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(torch.diag(torch.tensor([1.0, -1, -1, 1])))

        # camera coord matrix
        def cam_matrix(cam_d):
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, cam_d],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )

        cam_mat = torch.tensor(cam_matrix(camera_d))

        # merge
        return torch.matmul(ndc_proj, cam_mat)

    def get_identity(self, _id):
        assert _id.ndim == 2 and _id.shape[1] == 80
        N = _id.shape[0]
        delta_id = torch.matmul(self._id_base, _id.t())  # type: ignore
        iden = self.mean_pos + delta_id
        return iden.permute(1, 0).view(N, -1, 3)

    def get_verts(self, _id, exp):
        N = len(_id)
        delta_id = torch.matmul(self._id_base, _id.t())  # type: ignore
        delta_ex = torch.matmul(self.exp_base, exp.t())  # type: ignore
        v = self.mean_pos + delta_id + delta_ex
        return v.permute(1, 0).view(N, -1, 3)

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(angles.device)
        zeros = torch.zeros([batch_size, 1]).to(angles.device)
        x, y, z = (
            angles[:, :1],
            angles[:, 1:2],
            angles[:, 2:],
        )

        rot_x = torch.cat(
            [ones, zeros, zeros, zeros, torch.cos(x), -torch.sin(x), zeros, torch.sin(x), torch.cos(x)], dim=1
        ).reshape([batch_size, 3, 3])

        rot_y = torch.cat(
            [torch.cos(y), zeros, torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros, torch.cos(y)], dim=1
        ).reshape([batch_size, 3, 3])

        rot_z = torch.cat(
            [torch.cos(z), -torch.sin(z), zeros, torch.sin(z), torch.cos(z), zeros, zeros, zeros, ones], dim=1
        ).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def render(self, verts, angle, translation, trans_params, recenter=True, proj_name="tf", uv_name="official"):
        # id_coeff = coeff[:,:80] #identity
        # ex_coeff = coeff[:,80:144] #expression
        # tex_coeff = coeff[:,144:224] #texture
        # angles = coeff[:,224:227] #euler angles for pose
        # gamma = coeff[:,227:254] #lighting
        # translation = coeff[:,254:257] #translation

        if recenter:
            verts = verts - self.center
        rotat = self.compute_rotation(angle)
        verts = verts @ rotat + translation.unsqueeze(1)

        # to homo
        verts = F.pad(verts, (0, 1), "constant", value=1)

        # to camera
        assert proj_name in ["tf", "torch"]
        verts_screen = verts @ getattr(self, f"proj_{proj_name}").t()  # type: ignore
        verts_screen[..., 1] *= -1

        # set uv
        assert uv_name in ["official", "ours"]
        self.rast.template.NO_uvcoords.data = getattr(self, f"NO_uvcoords_{uv_name}")
        # rasterize
        ret = self.rast(verts, verts_screen, pixel_attr_names=("uvs",))

        # process outputs
        mask = ret["mask_valid"]
        uvs = (ret["pixel_uvs"] + 1) / 2.0
        uvs = F.pad(uvs, (0, 1), "constant", value=0)
        uvs = torch.where(mask, uvs, torch.zeros_like(uvs))

        def _reverse(tensor, fill):
            new_list = []
            for bi in range(tensor.shape[0]):
                x = tensor[bi].detach().cpu().numpy()
                x, _ = reverse_transform(x, trans_params[bi])
                if fill:
                    x = fill_hole((x * 255).astype(np.uint8)).astype(np.float32) / 255.0
                new_list.append(x)
            return torch.tensor(new_list, dtype=tensor.dtype, device=tensor.device)

        uvs = _reverse(uvs, False)
        mask = _reverse(mask.float(), True)
        return uvs.permute(0, 3, 1, 2)[:, :2], mask.permute(0, 3, 1, 2)


def fill_hole(trans):
    im_th = trans[..., 0]
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)  # type: ignore
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # type: ignore
    im_out = im_th | im_floodfill_inv
    return im_out[:, :, None]


def reverse_transform(img, trans):
    assert img.shape[0] == 224
    assert img.shape[1] == 224
    target_size = img.shape[0]

    w0, h0, s, t0, t1 = trans
    t0, t1 = t0[0], t1[0]
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)

    x0 = (w / 2 - target_size / 2 + float((t0 - w0 / 2) * s)).astype(np.int32)
    y0 = (h / 2 - target_size / 2 + float((h0 / 2 - t1) * s)).astype(np.int32)

    def _uncrop(im):
        im = np.pad(im, [[0, 0], [x0, 0], [0, 0]]) if x0 >= 0 else im[:, -x0:]
        im = np.pad(im, [[y0, 0], [0, 0], [0, 0]]) if y0 >= 0 else im[-y0:]
        im = np.pad(im, [[0, 0], [0, w - im.shape[1]], [0, 0]]) if w >= im.shape[1] else im[:, :w]
        im = np.pad(im, [[0, h - im.shape[0]], [0, 0], [0, 0]]) if h >= im.shape[0] else im[:h]
        return im

    def _resize(im):
        im = cv2.resize(im, (w0, h0), interpolation=cv2.INTER_CUBIC)
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
        return im

    mask = np.full_like(img, 1.0)

    img = _resize(_uncrop(img))
    mask = _resize(_uncrop(mask))
    mask[mask < 1.0] = 0
    return img, mask


bfm_model = BFM()
