import logging
import os
from typing import List, Tuple

import torch.nn as nn

from assets import ASSETS_ROOT
from src.modules.flame_mesh_cropper import FlameMeshCropper
from src.modules.layers import MLP
from src.modules.meshae import MeshConvEncoder

log = logging.getLogger("EncoderOffsets")


def build_encoder_offsets(config) -> Tuple[nn.Module, int]:
    if config.using == "m2i":
        from src.modules.mesh_image import MeshToImage

        m = _MeshToImageEncoder(config)
    elif config.using == "conv":
        m = _MeshConvEncoder(config)
    else:
        raise NotImplementedError()
    return m, m.out_channels


class _ImageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, first_channels, A):
        super().__init__()
        assert A >= 64

        # fmt: off
        def conv_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, norm=True, actv=True):
            """Returns downsampling layers of each discriminator block"""
            layers: List[nn.Module] = [nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if norm: layers.append(nn.InstanceNorm2d(out_filters))
            if actv: layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        # fmt: on

        inp_chs, out_chs, nfc = in_channels, out_channels, first_channels
        lys = []
        lys.extend(conv_block(inp_chs, nfc * 1, norm=False))
        lys.extend(conv_block(nfc * 1, nfc * 2))
        lys.extend(conv_block(nfc * 2, nfc * 4))
        lys.extend(conv_block(nfc * 4, nfc * 8))
        L = A // 64
        while L >= 2:
            L = L // 2
            lys.extend(conv_block(nfc * 8, nfc * 8))
        lys.extend(conv_block(nfc * 8, out_chs, stride=1, padding=0, norm=False, actv=False))
        self.main = nn.Sequential(*lys)
        self.out_channels = out_chs

    def forward(self, x):
        assert x.ndim == 4
        z = self.main(x)
        assert z.shape[2] == 1
        assert z.shape[3] == 1
        return z.view(z.shape[0], -1)


class _MeshToImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        template_fpath = os.path.join(ASSETS_ROOT, "selection", f"{config.part}.obj")
        self.main = nn.Sequential(
            FlameMeshCropper(config.part),
            MeshToImage(config.m2i.mode, config.m2i.image_size, template_fpath=template_fpath),
            _ImageEncoder(3, config.out_channels, config.m2i.nfc, config.m2i.image_size),
        )
        self.out_channels = config.out_channels
        log.info(f"encode mesh {config.part} (using m2i {config.m2i.mode}, {config.m2i.image_size})")

    def forward(self, x):
        assert x.ndim == 3
        x = self.main(x)
        return x


class _MeshConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        template_fpath = os.path.join(ASSETS_ROOT, "selection", f"{config.part}.obj")
        self.crop = FlameMeshCropper(config.part)
        self.main = MeshConvEncoder(config.mesh_conv, template_fpath)
        self.proj = MLP(self.main.out_channels, config.out_channels, activation="lrelu0.2")
        self.out_channels = config.out_channels
        log.info(f"encode mesh {config.part} (using {config.mesh_conv.conv_type}, {config.mesh_conv.ds_factors})")

    def forward(self, x):
        assert x.ndim == 3
        x = self.crop(x)  # select part
        x = self.main(x)  # main conv
        x = self.proj(x)  # proj
        return x
