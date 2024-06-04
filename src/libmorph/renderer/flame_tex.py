import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("FLAME")


class FLAMETex(nn.Module):
    """
    FLAME texture:
    https://github.com/TimoBolkart/TF_FLAME/blob/ade0ab152300ec5f0e8555d6765411555c5ed43d/sample_texture.py#L64
    FLAME texture converted from BFM:
    https://github.com/TimoBolkart/BFM_to_FLAME
    """

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == "BFM":
            mu_key = "MU"
            pc_key = "PC"
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == "FLAME":
            mu_key = "mean"
            pc_key = "tex_dir"
            n_pc = 200
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.0
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.0
        else:
            raise NotImplementedError(f"texture type {config.tex_type} not exist!")

        n_tex = config.n_tex
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        log.info("Creating the FLAMETex {} {} components".format(config.tex_type, n_tex))
        self.n_tex = n_tex
        self.register_buffer("texture_mean", texture_mean)
        self.register_buffer("texture_basis", texture_basis)

    def forward(self, texcode=None):
        """
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        """

        if texcode is None:
            texcode = torch.zeros((1, self.n_tex), dtype=torch.float32, device=self.texture_mean.device)

        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture
