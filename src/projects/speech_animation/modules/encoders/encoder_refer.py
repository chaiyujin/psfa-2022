import os
from typing import Tuple

import torch
import torch.nn as nn

import assets
from assets import ASSETS_ROOT
from src.modules.layers import MLP


def build_encoder_refxy(config) -> Tuple[nn.Module, int]:
    if config.using == "mlp":
        m = _MLPEncoder(config)
    elif config.using == "m2i":
        m = EncoderReferenceXY(config)
    else:
        raise NotImplementedError()
    return m, m.out_channels


class _MLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        out_ch = config.out_channels
        self.main = MLP(23 * 2, [out_ch * 2, out_ch * 2, out_ch], last_activation="identity")
        self.out_channels = out_ch

    def forward(self, x):
        assert x.ndim == 3
        x = x.view(x.shape[0], -1)
        z = self.main(x)
        return z


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, first_channels):
        super().__init__()

        # fmt: off
        def conv_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, norm=True, actv=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if norm: layers.append(nn.InstanceNorm2d(out_filters))
            if actv: layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        # fmt: on

        inp_chs, out_chs, nfc = in_channels, out_channels, first_channels
        self.main = nn.Sequential(
            *conv_block(inp_chs, nfc * 1, norm=False),
            *conv_block(nfc * 1, nfc * 2),
            *conv_block(nfc * 2, nfc * 4),
            *conv_block(nfc * 4, nfc * 8),
            *conv_block(nfc * 8, out_chs, stride=1, padding=0, norm=False, actv=False),
        )
        self.out_channels = out_chs

    def forward(self, x):
        assert x.ndim == 4
        z = self.main(x)
        assert z.shape[2] == 1 and z.shape[3] == 1, "output has wrong shape: {}".format(z.shape)
        return z[:, :, 0, 0]


class EncoderReferenceXY(nn.Module):
    def __init__(self, config):
        super().__init__()
        from src.modules.mesh_image import MeshToImage

        self.main = nn.Sequential(
            FlameMeshCropper(config.part),
            SelectXYZ([0, 1]),
            MeshToImage(**config.m2i, template_fpath=os.path.join(ASSETS_ROOT, "selection", f"{config.part}.obj")),
            _ImageEncoder(2, config.out_channels, config.nfc),
        )
        self.out_channels = config.out_channels

    def forward(self, x):
        assert x.ndim == 3
        x = self.main(x)
        return x


class FlameMeshCropper(nn.Module):
    def __init__(self, part):
        super().__init__()
        self.vidx = getattr(assets, f"{part.upper()}_VIDX")

    def forward(self, x):
        assert x.shape[-2] == 5023
        return x[..., self.vidx, :]

    def reverse(self, x):
        shape = list(x.shape)
        shape[-2] = 5023
        y = torch.zeros(shape, device=x.device, dtype=x.dtype)
        y[..., self.vidx, :] = x
        return y


class SelectXYZ(nn.Module):
    XYZ_INDEX = dict(x=0, y=1, z=2)

    def __init__(self, index):
        super().__init__()

        def to_positive(x):
            while x < 0:
                x += 3
            assert 0 <= x <= 2
            return x

        if isinstance(index, int):
            index = [index]
        elif isinstance(index, str):
            index = [self.XYZ_INDEX[v] for v in index]
        self.index = [to_positive(x) for x in index]

    def forward(self, x):
        assert x.shape[-1] == 3
        return x[..., self.index]

    def reverse(self, part):
        assert part.shape[-1] == len(self.index)
        shape = list(part.shape)
        shape[-1] = 3
        xyz = torch.zeros(shape, dtype=part.dtype, device=part.device)
        xyz[..., self.index] = part
        return xyz
