from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.activation import LeakyReLU
from torch.nn.parameter import Parameter

from assets import get_selection_obj
from src.modules.mesh_conv import build_mesh_conv_blocks


def _load_tex3(tex, load_from, load_resize, load_offset, A):
    a = int(A * load_resize)
    assert a <= A
    x = int((A - a) / 2 + A * load_offset[0])
    y = int((A - a) / 2 + A * load_offset[1])

    im = cv2.imread(load_from)[..., [2, 1, 0]]  # rgb
    im = cv2.resize(im, (a, a)).astype(np.float32) / 255.0
    im = im * 2.0 - 1.0
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    tex3ch = torch.zeros_like(tex[:, :3])
    tex3ch[:, :, y : y + a, x : x + a] = im_tensor
    return tex3ch


class StaticTextures(nn.Module):
    def __init__(self, config, load_from=None, load_resize=1, load_offset=(0, 0)):
        super().__init__()
        assert len(config.clip_sources) == 1

        tex = torch.randn(1, config.tex_features, config.tex_dim, config.tex_dim, dtype=torch.float32)
        if load_from is not None:
            tex = _load_tex3(tex, load_from, load_resize, load_offset, config.tex_dim)
        self.register_parameter("tex_data", Parameter(tex))

    def forward(self, *args):
        return self.tex_data


class DynamicTextures(nn.Module):
    def __init__(
        self,
        config,
        d_tex_inter=None,
        n_filters=1,
        kernel_size=3,
        warping=False,
        use_rotat=None,
    ):
        super().__init__()
        assert len(config.clip_sources) == 1
        assert n_filters >= 1

        if d_tex_inter is None:
            d_tex_inter = config.tex_features
        tex = torch.randn(1, d_tex_inter, config.tex_dim, config.tex_dim, dtype=torch.float32)
        self.register_parameter("tex_data", Parameter(tex))

        # face vertex encoder
        self.face_encoder = build_mesh_conv_blocks(get_selection_obj("face2"), "encoder", **config.mesh_conv)
        d_hid = config.mesh_conv.latent_channels
        d_out = self.face_encoder.blocks[-1].n_verts_out * self.face_encoder.out_channels  # type: ignore
        d_flt = config.tex_features * d_tex_inter + d_tex_inter * d_tex_inter * (n_filters - 1)
        # (optional) rotation condition
        self.use_rotat = use_rotat
        if use_rotat is not None:
            # use fourier feature mapping
            mapping_size = int(self.face_encoder.out_channels)  # type: ignore
            self.register_buffer("B", torch.randn((use_rotat, mapping_size), dtype=torch.float32))
            d_out += mapping_size * 2

        # fmt: off
        self.proj_w = nn.Sequential(
            nn.Linear(d_out, d_hid), nn.LeakyReLU(0.2),
            nn.Linear(d_hid, d_hid), nn.LeakyReLU(0.2),
            nn.Linear(d_hid, d_flt * (kernel_size ** 2)),
        )
        if warping:
            self.proj_grid = nn.Sequential(
                nn.Linear(d_out, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 2 * ((config.tex_dim//4) ** 2)), nn.Tanh()
            )
        # fmt: on

        self.tex_size = config.tex_dim
        self.d_tex = config.tex_features
        self.d_tex_inter = d_tex_inter
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.ksz_sqr = kernel_size**2

    def forward(self, v_face, rot: Optional[Tensor] = None):
        z_face = self.face_encoder(v_face)
        z_face = z_face.view(*z_face.shape[:-2], -1)
        if self.use_rotat is not None and self.use_rotat > 0:
            assert rot is not None
            # print("given rotation", rot.shape)
            rot_proj = torch.matmul(2.0 * np.pi * rot, self.B)  # type: ignore
            z_rot = torch.cat([torch.sin(rot_proj), torch.cos(rot_proj)], dim=-1)
            z_face = torch.cat((z_face, z_rot), dim=-1)
        weights = self.proj_w(z_face)
        bsz = weights.shape[0]

        textures: Tensor = self.tex_data  # type: ignore
        if hasattr(self, "tex_first3ch"):
            textures = torch.cat((self.tex_first3ch, textures), dim=1)  # type: ignore
        assert textures.shape[0] == 1

        if hasattr(self, "proj_grid"):
            grid = self.proj_grid(z_face)
            grid = grid.view(bsz, 2, self.tex_size // 4, self.tex_size // 4)
            grid = F.upsample(grid, mode="bilinear", scale_factor=4, align_corners=False)
            grid = grid.permute(0, 2, 3, 1)
            textures = textures.expand(bsz, -1, -1, -1)
            textures = F.grid_sample(textures, grid, mode="bilinear", align_corners=False)

        conv_filters, s = [], 0
        for i_filter in range(self.n_filters):
            if i_filter + 1 < self.n_filters:
                n = self.d_tex_inter * self.d_tex_inter * self.ksz_sqr
                flt = weights[:, s : s + n]
                flt = flt.view(bsz, self.d_tex_inter, self.d_tex_inter, self.kernel_size, self.kernel_size)
            else:
                n = self.d_tex * self.d_tex_inter * self.ksz_sqr
                flt = weights[:, s : s + n]
                flt = flt.view(bsz, self.d_tex, self.d_tex_inter, self.kernel_size, self.kernel_size)
            conv_filters.append(flt)
            s += n
        assert s == weights.shape[1]
        assert self.n_filters == len(conv_filters)

        textures = self._filter_textures(textures, conv_filters, bsz)
        return textures

    def _filter_textures(self, tex, conv_filters, bsz):
        pad_l = (self.kernel_size - 1) // 2
        pad_r = (self.kernel_size - 1) - pad_l

        def _pad(x):
            return F.pad(x, (pad_l, pad_r, pad_l, pad_r))

        tex_list = []
        for bi in range(bsz):
            x = tex if tex.shape[0] == 1 else tex[bi : bi + 1]
            for fi, filters in enumerate(conv_filters):
                weight = filters[bi]
                x = F.conv2d(_pad(x), weight)
                if fi + 1 < self.n_filters:
                    x = F.leaky_relu(x, 0.2, inplace=True)
                #     print("lrelu0.2")
                # print(bi, x.shape, weight.shape)
            tex_list.append(x)
        return torch.cat(tex_list, dim=0)
