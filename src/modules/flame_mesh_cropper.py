import torch
import torch.nn as nn

import assets


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
