from copy import deepcopy
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.data.mesh.torch import pool_bvc
from src.engine.ops import parse_activation

from .spiralconv import SpiralConv
from .utils import preprocess_template

# -------------------------------------------------------------------------------------------------------------------- #
#                                                        Blocks                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


class SpiralPool(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x):
        return pool_bvc(x, self.transform)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, list(self.transform.shape))


class SpiralEnblock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        indices,
        down_transform,
        activation="identity",
    ):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, hidden_channels, indices)
        self.actv = parse_activation(activation)
        self.pool = SpiralPool(down_transform)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = self.actv(self.conv(x))
        x = self.pool(x)
        return x


class SpiralDeblock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        indices,
        up_transform,
        activation="identity",
        act_kwargs=dict(),
    ):
        super(SpiralDeblock, self).__init__()
        self.pool = SpiralPool(up_transform)
        self.actv = parse_activation(activation)
        self.conv = SpiralConv(in_channels, hidden_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x):
        x = self.pool(x)
        x = self.actv(self.conv(x))
        return x


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        Modules                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class SpiralNet(nn.Module):
    def __init__(
        self,
        template_path: str,
        in_channels: int,
        hidden_channels: Tuple[int, Tuple[int]],
        latent_channels: int,
        seq_lengths: Tuple[int, ...] = (
            9,
            9,
            9,
        ),
        dilations: Tuple[int, ...] = (
            1,
            1,
            1,
        ),
        ds_factors: Tuple[int, ...] = (
            4,
            4,
            4,
        ),
        layers_activation: str = "elu",
        latent_activation: str = "identity",
    ):
        super(SpiralNet, self).__init__()

        # -> to tuple
        hidden_channels = (hidden_channels,) if isinstance(hidden_channels, int) else hidden_channels

        # we have to manually reversed some hparams in decoder
        self.template_path = template_path
        self.encoder = SpiralEncoder(
            template_path=template_path,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            seq_lengths=seq_lengths,
            dilations=dilations,
            ds_factors=ds_factors,
            layers_activation=layers_activation,
            latent_activation=latent_activation,
        )
        self.decoder = SpiralDecoder(
            template_path=template_path,
            latent_channels=latent_channels,
            hidden_channels=tuple(reversed(hidden_channels)),
            out_channels=in_channels,
            seq_lengths=seq_lengths,
            dilations=dilations,
            ds_factors=ds_factors,
            layers_activation=layers_activation,
        )
        self.latent_channels = self.encoder.latent_channels

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z


class SpiralEncoder(nn.Module):
    def __init__(
        self,
        template_path: str,
        in_channels: int,
        hidden_channels: Tuple[int, Tuple[int]],
        latent_channels: int,
        seq_lengths: Tuple[int, ...] = (
            9,
            9,
            9,
        ),
        dilations: Tuple[int, ...] = (
            1,
            1,
            1,
        ),
        ds_factors: Tuple[int, ...] = (
            4,
            4,
            4,
        ),
        layers_activation: str = "elu",
        latent_activation: str = "identity",
    ):
        super(SpiralEncoder, self).__init__()
        # init from template
        self.template_path = template_path
        self.spiral_indices, self.down_transform, _ = preprocess_template(
            template_path, tuple(seq_lengths), tuple(dilations), tuple(ds_factors)
        )

        # get size
        self.n_verts = self.down_transform[-1].shape[0]
        self.in_channels = in_channels
        self.hidden_channels = (hidden_channels,) if isinstance(hidden_channels, int) else tuple(hidden_channels)
        self.latent_channels = latent_channels
        # check types
        assert isinstance(self.in_channels, int)
        assert isinstance(self.hidden_channels, (list, tuple))
        assert isinstance(self.latent_channels, int)
        assert len(self.hidden_channels) == len(ds_factors)

        # encoder
        inp_list = (self.in_channels,) + tuple(self.hidden_channels[:-1])
        out_list = tuple(self.hidden_channels)
        self.encoder_layers = nn.ModuleList()
        for idx, (inp, out) in enumerate(zip(inp_list, out_list)):
            self.encoder_layers.append(
                SpiralEnblock(
                    inp,
                    out,
                    self.spiral_indices[idx],
                    self.down_transform[idx],
                    activation=layers_activation,
                )
            )
        self.encoder_latent_projector = nn.Linear(self.n_verts * self.hidden_channels[-1], self.latent_channels)

        # latent activation
        self.latent_activation = parse_activation(latent_activation)

        # init parameters
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encode(self, x):
        # encoder layers
        assert x.shape[-1] == self.in_channels
        # print('spiral_encoder ~ input:', x.shape)
        for layer in self.encoder_layers:
            x = layer(x)
            # print('spiral_encoder ~ layer:', x.shape)

        # project into latent
        x = x.view(x.shape[0], self.n_verts * self.hidden_channels[-1])
        z = self.encoder_latent_projector(x)
        # print('spiral_encoder ~ flatten:', x.shape)
        # print('spiral_encoder ~ project:', z.shape)

        # activate latent
        z = self.latent_activation(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        return z


class SpiralDecoder(nn.Module):
    def __init__(
        self,
        template_path: str,
        latent_channels: Tuple[int, Tuple[int]],
        hidden_channels: Tuple[int, Tuple[int]],
        out_channels: int,
        seq_lengths: Tuple[int, ...] = (
            9,
            9,
            9,
        ),
        dilations: Tuple[int, ...] = (
            1,
            1,
            1,
        ),
        ds_factors: Tuple[int, ...] = (
            4,
            4,
            4,
        ),
        layers_activation: str = "elu",
    ):
        super(SpiralDecoder, self).__init__()
        # init from template
        self.template_path = template_path
        self.spiral_indices, _, self.up_transform = preprocess_template(
            template_path, tuple(seq_lengths), tuple(dilations), tuple(ds_factors)
        )
        # get size
        self.n_verts = self.up_transform[-1].shape[1]
        self.latent_channels = (latent_channels,) if isinstance(latent_channels, int) else tuple(latent_channels)
        self.hidden_channels = (hidden_channels,) if isinstance(hidden_channels, int) else tuple(hidden_channels)
        self.out_channels = out_channels
        # check types
        assert isinstance(self.latent_channels, (list, tuple))
        assert isinstance(self.hidden_channels, (list, tuple))
        assert isinstance(self.out_channels, int)
        assert len(self.hidden_channels) == len(ds_factors)
        self.n_latents = len(self.latent_channels)

        # projector
        # - split hidden_channels 0 into several parts
        C = sum(self.latent_channels)
        full_size = self.hidden_channels[0]
        out_sizes = [int(x * full_size / C) for x in self.latent_channels]
        L = full_size - sum(out_sizes)
        assert L >= 0
        idx = 0
        while L > 0:
            out_sizes[idx] += 1
            idx += 1
            L -= 1
        assert all(x > 0 for x in out_sizes), "The first element in hiddent_channels is too small!"
        # - build projectors
        self.projectors = nn.ModuleList(
            [nn.Linear(z, x * self.n_verts) for (z, x) in zip(self.latent_channels, out_sizes)]
        )

        # decoder
        inp_list = tuple(self.hidden_channels)
        out_list = tuple(self.hidden_channels[1:]) + (self.out_channels,)
        self.decoder_layers = nn.ModuleList()
        for idx, (inp, out) in enumerate(zip(inp_list, out_list)):
            self.decoder_layers.append(
                SpiralDeblock(
                    inp,
                    out,
                    self.spiral_indices[-idx - 1],
                    self.up_transform[-idx - 1],
                    activation=layers_activation if idx + 1 < len(out_list) else "identity",
                )
            )

        # init parameters
        self.reset_parameters()

    @property
    def decoded_channels(self):
        return self.out_channels

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def decode(self, z_list):
        # check number of latents
        if torch.is_tensor(z_list):
            z_list = (z_list,)
        assert len(z_list) == self.n_latents, "Given {} latents, but should be {}".format(len(z_list), self.n_latents)

        # project
        x_list = [projector(z).view(z.shape[0], self.n_verts, -1) for z, projector in zip(z_list, self.projectors)]
        x = torch.cat(x_list, dim=-1)

        # decoder
        # print('spiral_decoder ~ input:', x.shape)
        for layer in self.decoder_layers:
            x = layer(x)
            # print('spiral_decoder ~ layer:', x.shape)

        return x

    def forward(self, z_or_list):
        return self.decode(z_or_list)


# simple tests
if __name__ == "__main__":
    import os

    from src.utils.data.mesh import io as meshio

    template_path = os.path.expanduser("~/assets/3DMM-FLAME/Processed/templates/PART_FULL/TMPL.obj")
    x, _, _ = meshio.load_mesh(template_path)
    x = torch.from_numpy(x).unsqueeze(0).repeat(16, 1, 1).cuda()
    print(x.shape)

    spiralnet = SpiralNet(
        template_path,
        3,
        (16, 32, 32),
        8,
        (9, 9, 9),
        (1, 1, 1),
        (4, 4, 4),
    ).cuda()

    y, z = spiralnet(x)
    print(x.shape, y.shape, z.shape)
