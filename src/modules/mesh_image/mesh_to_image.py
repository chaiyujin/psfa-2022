import os
import pickle

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from assets import ASSETS_ROOT
from src.mm_fitting.libmorph.renderer.template import MeshTemplate

from .rast_2d import cython_rasterize_2d


class MeshToImage(nn.Module):
    def __init__(self, mode, image_size, template_fpath, debug=False):
        super().__init__()
        assert mode in ["rast", "nearest"]
        self.mode = mode
        self.image_size = image_size
        self.template = MeshTemplate(template_fpath)
        self.debug = debug
        self._tag = os.path.basename(os.path.splitext(template_fpath)[0])

        # find correspondance uv <- verts
        mapping = dict()
        tris_pos = self.template.triangles.detach().cpu().numpy()[0]
        tris_uvs = self.template.uv_triangles.detach().cpu().numpy()[0]
        for tri_pos, tri_uvs in zip(tris_pos, tris_uvs):
            for i_pos, i_uvs in zip(tri_pos, tri_uvs):
                if i_uvs in mapping:
                    assert (
                        mapping[i_uvs] == i_pos
                    ), "Given template {} cannot biject between uv coordinates and verts".format(template_fpath)
                mapping[i_uvs] = i_pos
        self.indices = [mapping[i] for i in range(len(mapping))]
        # reverse indices
        rev_map = {j: i for i, j in mapping.items()}
        self.indices_rev = [rev_map[i] for i in range(len(rev_map))]

        # * Get static UV vertices
        uv_no = self.template.NO_uvcoords.detach().clone()[0]
        xmin, xmax = uv_no[..., 0].min(), uv_no[..., 0].max()
        ymin, ymax = uv_no[..., 1].min(), uv_no[..., 1].max()
        # to center
        uv_no[..., 0:1] = (uv_no[..., 0:1] - xmin) / (xmax - xmin)
        uv_no[..., 1:2] = (uv_no[..., 1:2] - ymin) / (ymax - ymin)
        # shrink
        uv_no = (uv_no - 0.5) * 0.95 + 0.5
        # round
        uv_pts = torch.round(uv_no * (image_size - 1))

        # * Rasterize
        tmpl_name = "-".join(os.path.splitext(template_fpath)[0].split("/")[-2:])
        if self.mode == "rast":
            cache_path = os.path.join(ASSETS_ROOT, ".mesh_to_image", f"rast-{tmpl_name}-{image_size}.npy")
            if not os.path.exists(cache_path):
                pts = uv_pts.numpy().astype(np.int32)
                rast_out = cython_rasterize_2d(pts, (image_size, image_size), self.template.uv_triangles[0].numpy())
                rast_out = nearest_fill(rast_out, lambda x: x[-1] > 0)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, rast_out)
            rast_out = np.load(cache_path)
            self.register_buffer("rast_out", torch.tensor(rast_out[None, ...]))
            self.register_buffer("grid", uv_pts[None, :, None, :] / (image_size - 1) * 2 - 1)
            # if self.debug:
            #     import cv2
            #     cv2.imshow("[MeshToImage]: rasterized", rast_out[..., :3])
            #     cv2.waitKey(0)
        elif self.mode == "nearest":
            cache_path = os.path.join(ASSETS_ROOT, ".mesh_to_image", f"nearest-{tmpl_name}-{image_size}.pkg")
            if not os.path.exists(cache_path):
                i2v_idx = []
                v2i_idx = np.full((image_size, image_size), fill_value=-1, dtype=np.int64)
                for i, xy in enumerate(uv_pts):
                    x, y = int(xy[0]), int(xy[1])
                    idx = y * image_size + x
                    i2v_idx.append(idx)
                    v2i_idx[y, x] = i
                # fill v2i_idx using nearest neighbour
                v2i_idx = nearest_fill(v2i_idx, lambda x: x >= 0)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as fp:
                    pickle.dump(dict(i2v_idx=i2v_idx, v2i_idx=v2i_idx), fp)
            with open(cache_path, "rb") as fp:
                data = pickle.load(fp)
                i2v_idx = data["i2v_idx"]
                v2i_idx = data["v2i_idx"]
            self.register_buffer("v2i_idx", torch.tensor(v2i_idx, dtype=torch.long))
            self.register_buffer("i2v_idx", torch.tensor(i2v_idx, dtype=torch.long))
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")

        # self._dump_debug_view()

    def _dump_debug_view(self):
        verts = self.template.vertices
        img = self(verts)
        # debug vis
        vis_img = img[0].permute(1, 2, 0).detach().cpu().numpy()
        for ch in range(vis_img.shape[2]):
            vmin = vis_img[..., ch].min()
            vmax = vis_img[..., ch].max()
            vis_img[..., ch] = (vis_img[..., ch] - vmin) / (vmax - vmin)
        vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)
        import cv2

        cv2.imwrite(f"m2i-{self._tag}.png", vis_img)
        # cv2.imshow("img", vis_img)
        # cv2.waitKey()

        from src.data.mesh.io import save_obj

        recv = self.reverse(img)
        recv = recv[0].detach().cpu().numpy()
        # assert (recv[0, mesh_converter.vidx_from_full] == verts[0, mesh_converter.vidx_from_full]).all()
        save_obj(f"m2i_rec-{self._tag}.obj", recv, self.template.triangles[0])

    def forward(self, vertex_values):
        # check shape and select from full
        bsz = vertex_values.shape[0]
        assert vertex_values.ndim == 3, "invalid shape: {}".format(vertex_values.shape)
        assert vertex_values.shape[-2] == self.template.vertices.shape[-2]
        # re-arrange into uv indices
        uv_values = vertex_values[..., self.indices, :].contiguous()

        if self.mode == "rast":
            # interpolate
            rast = self.rast_out.expand(bsz, -1, -1, -1).contiguous()
            image, _ = dr.interpolate(uv_values, rast, self.template.uv_triangles[0])
        elif self.mode == "nearest":
            image = uv_values[..., self.v2i_idx, :]
        else:
            raise NotImplementedError()

        # import cv2
        # dbg_img = image[0].detach().cpu().numpy()
        # if dbg_img.shape[-1] == 2:
        #     dbg_img = np.pad(dbg_img, [[0,0], [0,0], [0,1]], "constant")
        # cv2.imshow('m2i', cv2.resize(dbg_img * 100, (256, 256)))
        # cv2.waitKey(1)

        return image.permute(0, 3, 1, 2)

    def reverse(self, image):
        assert image.ndim == 4 and image.shape[-2] == self.image_size and image.shape[-1] == self.image_size

        if self.mode == "rast":
            # fetch from uv
            grid = self.grid.expand(image.shape[0], -1, -1, -1)
            uv_values = F.grid_sample(image, grid, mode="bilinear", align_corners=True).squeeze(-1).permute(0, 2, 1)
        elif self.mode == "nearest":
            flatten = image.view(image.shape[0], image.shape[1], -1)
            uv_values = flatten[:, :, self.i2v_idx].permute(0, 2, 1)
        else:
            raise NotImplementedError()

        vertex_values = uv_values[..., self.indices_rev, :]
        return vertex_values


def nearest_fill(data, valid_fn):
    org = data.copy()
    H, W = data.shape[:2]
    # store all valid points
    valid_xy = []
    for y in range(H):
        for x in range(W):
            if valid_fn(data[y, x]):
                valid_xy.append((x, y))
    # for each value, find the closest valid
    for y in tqdm(range(H), desc="Nearest Fill"):
        for x in range(W):
            # original valid
            if valid_fn(data[y, x]):
                continue
            min_dist = H * W
            for s, t in valid_xy:
                dist = (s - x) ** 2 + (t - y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    data[y, x] = org[t, s]
    return data
