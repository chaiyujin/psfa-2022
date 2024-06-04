from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .lights import add_lights


class SH9Shader(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_attr_requirements = ("vertices", "normals", "uvs")

    def forward(
        self,
        rast_dict: Dict[str, Any],
        textures: Optional[Tensor] = None,
        lights: Optional[Tensor] = None,
    ):

        mask_valid = rast_dict["mask_valid"]

        # images
        image_albedos = None
        image_shading = None
        images = None

        # albedo texture
        if textures is not None:
            grid = rast_dict["pixel_uvs"]  # N,H,W,2
            grid = torch.where(mask_valid, grid, torch.full_like(grid, -2))
            # should be in shape: N,C,H,W
            if textures.shape[-1] in [1, 3, 4]:
                textures = textures.permute(0, 3, 1, 2)
            if textures.shape[0] != grid.shape[0]:
                assert textures.shape[0] == 1
                textures = textures.expand(grid.shape[0], -1, -1, -1)
            image_albedos = F.grid_sample(textures, grid, align_corners=False).permute(0, 2, 3, 1)

            # use default albedos as final output
            images = image_albedos

        # lights
        if lights is not None:
            image_shading = add_lights("sh9", lights, **rast_dict)
            images = images * image_shading

        return dict(
            images=images,
            image_albedos=image_albedos,
            image_shading=image_shading,
        )
