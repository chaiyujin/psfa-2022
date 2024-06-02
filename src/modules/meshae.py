from copy import deepcopy

import torch
import torch.nn as nn
from omegaconf import open_dict

from src.engine.logging import get_logger
from src.engine.misc.run_once import run_once
from src.engine.ops import parse_activation, parse_norm
from src.modules import AutoPadding
from src.modules.layers import MLP, Conv2d, Linear
from src.modules.mesh_conv import build_mesh_conv_blocks

log = get_logger(__name__)

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                              Mesh Conv Auto-Encoder                                              * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def _get_mesh_conv_kwargs(config):
    assert "last_activation" not in config
    kwargs = deepcopy(config)
    with open_dict(kwargs):
        for key in (
            "in_channels",
            "out_channels",
            "hidden_channels",
            "latent_channels",
            "latent_activation",
            "output_activation",
        ):
            if key in kwargs:
                kwargs.pop(key)
    return kwargs


class MeshConvEncoder(nn.Module):
    def __init__(self, config, template_path):
        super().__init__()
        kwargs = _get_mesh_conv_kwargs(config)

        # build encoder blocks
        self._conv = build_mesh_conv_blocks(
            block_type="encoder",
            template_path=template_path,
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            **kwargs,
        )
        self._n_verts = self._conv.blocks[-1].n_verts_out

        # encoder projection
        self._proj = MLP(
            in_channels=self._n_verts * config.hidden_channels[-1],
            out_channels=[config.latent_channels * 2, config.latent_channels * 2, config.latent_channels],
            norm_method=config.norm_method,
            activation=config.activation,
            last_activation=config.latent_activation,
        )

    @property
    def out_channels(self):
        return self._proj.out_channels

    def forward(self, x):
        assert x.ndim == 3
        x = self._conv(x)

        bsz = x.shape[0]
        z = x.view(bsz, -1)
        z = self._proj(z)

        return z


class MeshConvDecoder(nn.Module):
    def __init__(self, config, template_path):
        super().__init__()
        kwargs = _get_mesh_conv_kwargs(config)

        # build decoder blocks
        conv = build_mesh_conv_blocks(
            block_type="decoder",
            template_path=template_path,
            in_channels=config.hidden_channels[0],
            hidden_channels=tuple(config.hidden_channels[1:]) + (config.out_channels,),
            last_activation=config.output_activation,
            **kwargs,
        )
        self._n_verts = conv.blocks[0].n_verts_in

        # decoder projection
        self._proj = MLP(
            in_channels=config.latent_channels,
            out_channels=[
                config.latent_channels * 2,
                config.latent_channels * 2,
                self._n_verts * config.hidden_channels[0],
            ],
            norm_method=config.norm_method,
            activation=config.activation,
            last_activation=config.activation,
        )

        self._conv = conv

    def forward(self, z):
        assert z.ndim == 2
        bsz = z.shape[0]
        z = self._proj(z)

        x = z.view(bsz, self._n_verts, -1)
        x = self._conv(x)

        return x


class MeshConvAE(nn.Module):
    def __init__(self, config, template_path):
        super().__init__()

        enc_config = deepcopy(config)
        dec_config = deepcopy(config)

        with open_dict(dec_config):
            # reverse hidden channels for decoder
            dec_config.hidden_channels = tuple(reversed(dec_config.hidden_channels))
            # set out_channels if not given
            if dec_config.get("out_channels") is None:
                dec_config["out_channels"] = dec_config.in_channels

        self._encoder = MeshConvEncoder(enc_config, template_path)
        self._decoder = MeshConvDecoder(dec_config, template_path)

    def forward(self, x):
        assert x.ndim == 3
        z = self._encoder(x)
        y = self._decoder(z)

        self._debug_info()
        return y, z

    @run_once()
    def _debug_info(self):
        log.info(f"MeshAutoencoder:\n{self}")
