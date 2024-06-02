import logging
from typing import List, Tuple

import torch.nn as nn

from src.modules.attentions import create_self_atten
from src.modules.layers import LSTM, PositionalEncodingLNC

from .transformer import Transformer

logger = logging.getLogger(__name__)


def build_encoder_audio(hparams) -> Tuple[nn.Module, int]:
    # check if using
    name = hparams.using
    # alias
    if name in ["transformer"]:
        name = "xfmr"
    # build
    _build_dict = dict(
        conv=_ConvDeepSpeech,
        attn=_Attention,
        xfmr=_Transformer,
        conv_upper=_ConvUpper,
    )
    assert name in _build_dict, f"audio encoder is using unknown module '{name}'"
    m = _build_dict[name](hparams[name])
    return m, m.latent_channels  # type: ignore


class _TransposeWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self._layer(x)
        x = x.permute(0, 2, 1)
        return x


class _ConvDeepSpeech(nn.Module):
    def __init__(self, hparams):
        """Similar with convolution network used in NeuralVoicePuppetry
        (https://github.com/keetsky/NeuralVoicePuppetry)
        """
        super().__init__()

        # fmt: off
        def _conv(inp, out, ksz, stride, padding, actv=True, layer_norm=hparams.layer_norm, dropout=hparams.dropout):
            ret = [nn.Conv1d(inp, out, kernel_size=ksz, stride=stride, padding=padding)]
            if actv: ret.append(nn.LeakyReLU(0.2, inplace=True))
            if layer_norm: ret.append(_TransposeWrapper(nn.LayerNorm(out)))
            if dropout > 0: ret.append(nn.Dropout(dropout, inplace=True))
            return ret

        out = hparams.out_channels
        self.latent_channels = out

        layers = []
        layers.extend(_conv(29, 32, 3, 2, 1))  # 29 x 16 => 32 x 8
        layers.extend(_conv(32, 32, 3, 2, 1))  # 32 x 8  => 32 x 4
        layers.extend(_conv(32, 64, 3, 2, 1))  # 32 x 4  => 64 x 2
        layers.extend(_conv(64, out, 3, 2, 1, dropout=0, layer_norm=False))  # 64 x 2 => 64 x 1
        # fmt: on

        self._main = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        assert x.ndim == 4 and x.shape[1] in [16] and x.shape[2] == 29 and x.shape[3] == 1  # N, 16, 29, 1
        x = x.permute(0, 2, 1, 3).squeeze(3)  # N, C, L
        z = self._main(x)
        assert z.shape[2] == 1
        return z.squeeze(2)


class _Transformer(nn.Module):
    def __init__(self, opts):
        super().__init__()
        d_model = opts.d_model
        # proj
        self.in_channels = opts.get("in_channels", 29)
        self._proj_qry = nn.Linear(self.in_channels, d_model)
        self._proj_mem = nn.Linear(self.in_channels, d_model)
        # position encoding
        self._pos_enc = PositionalEncodingLNC(d_model, dropout=opts.dropout)
        # transformer
        self._transformer = Transformer(
            d_model,
            nhead=opts.n_head,
            num_encoder_layers=opts.n_enc_layers,
            num_decoder_layers=opts.n_dec_layers,
            dropout=opts.dropout,
            layer_norm=opts.layer_norm,
        )
        self.latent_channels = d_model

    def forward(self, x, **kwargs):
        assert x.ndim == 4
        assert x.shape[2] * x.shape[3] == self.in_channels
        x = x.view(x.shape[0], x.shape[1], -1)  # N, L, C
        x = x.permute(1, 0, 2)  # L, N, C

        q = self._proj_qry(x[x.shape[0] // 2]).unsqueeze(0)
        m = self._proj_mem(x)
        m = self._pos_enc(m)
        y = self._transformer(m, q)  # 1, N, C
        align = self._transformer.decoder.layers[0]._attn_weights
        assert y.shape[0] == 1
        assert y.ndim == 3

        if kwargs.get("align_dict") is not None:
            kwargs["align_dict"]["audio_attn"] = align.detach()

        z = y.squeeze(0)
        return z


class _Attention(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.in_channels = opts.get("in_channels", 29)
        # bilstm
        self._lstm = LSTM(
            self.in_channels,
            opts.memory_size // 2,
            num_layers=opts.n_layers,
            bias=False,
            batch_first=True,
            bidirectional=True,
        )
        self._norm = nn.LayerNorm(opts.memory_size)
        # attn
        self._attn = create_self_atten(**opts)
        self._qry_r = opts.query_radius
        self.latent_channels = opts.memory_size

    def forward(self, x, **kwargs):
        assert x.ndim == 4 and x.shape[2] * x.shape[3] == self.in_channels
        x = x.view(x.shape[0], x.shape[1], -1)  # N, L, C
        N, L = x.shape[:2]

        x, _ = self._lstm(x)
        x = self._norm(x)

        # return output and alignment
        stt = L // 2 - self._qry_r + 1
        end = L // 2 + self._qry_r
        qry = x[:, stt:end, :]
        z, align = self._attn(query=qry, key=x)

        assert z.ndim == 3 and z.shape[1] == 1
        z = z.squeeze(1)

        if kwargs.get("align_dict") is not None:
            kwargs["align_dict"]["audio_attn"] = align.detach()

        return z


class _ConvUpper(nn.Module):
    def __init__(self, hparams):
        """Similar with convolution network used in NeuralVoicePuppetry
        (https://github.com/keetsky/NeuralVoicePuppetry)
        """
        super().__init__()

        # fmt: off
        def _conv(inp, out, ksz, stride, dilation=1, actv=True, layer_norm=hparams.layer_norm, dropout=hparams.dropout):
            ret: List[nn.Module] = [nn.Conv1d(inp, out, kernel_size=ksz, stride=stride, dilation=dilation)]
            if actv: ret.append(nn.LeakyReLU(0.2, inplace=False))
            if layer_norm: ret.append(_TransposeWrapper(nn.LayerNorm(out)))
            if dropout > 0: ret.append(nn.Dropout(dropout, inplace=False))
            return ret

        out = hparams.out_channels
        self.latent_channels = out

        layers = []
        layers.extend(_conv(128, 32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  32, 2, 2))
        layers.extend(_conv(32,  out, 2, 2, dropout=0, layer_norm=False))  # 64 x 2 => 64 x 1
        # fmt: on

        self._main = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        assert x.ndim == 4 and x.shape[1] in [256] and x.shape[2] == 128 and x.shape[3] == 1  # N, L, C, 1
        x = x.permute(0, 2, 1, 3).squeeze(3)  # N, C, L
        z = self._main(x)
        assert z.shape[2] == 1
        return z.squeeze(2)
