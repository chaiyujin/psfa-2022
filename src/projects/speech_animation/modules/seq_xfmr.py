from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.engine import ops
from src.modules.layers import PositionalEncodingLNC


def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def sub_window_mask(sz: int, w: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = torch.ones(sz, sz)
    for i in range(sz):
        for j in range(i + 1, sz):
            mask[i, j] = 0
        for j in range(0, max(0, i - w + 1)):
            mask[i, j] = 0
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class SeqXFMR(nn.Module):
    def __init__(self, hparams, in_channels, z_style_channels, src_seq_frames, tgt_seq_frames, src_seq_pads):
        super().__init__()
        self.hparams = hparams
        self.in_channels = in_channels
        self.out_channels = hparams.d_model
        self.style_channels = z_style_channels
        self.win = hparams.win_size

        self.src_seq_frames = src_seq_frames
        self.tgt_seq_frames = tgt_seq_frames
        self.seq_pad = src_seq_pads[0]

        assert src_seq_frames == tgt_seq_frames
        assert src_seq_pads[0] == self.win - 1
        assert src_seq_pads[1] == 0
        self.register_buffer("mask", sub_window_mask(tgt_seq_frames + self.win - 1, self.win))

        self._build_encoder(hparams, in_channels)
        self._build_decoder(hparams, in_channels)
        # # create position encoding layer
        # self._pos_enc = PositionalEncodingLNC(hparams.d_model, dropout=hparams.dropout_posenc, max_len=120)
        # self._pos_dec = PositionalEncodingLNC(hparams.d_model, dropout=0, max_len=120)

    def _build_encoder(self, hparams, in_channels):
        # create projection linear
        self._project_input = ops.init_layer(
            nn.Linear(in_channels + self.style_channels, hparams.d_model), init_type="xavier_uniform"
        )
        # create the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparams.d_model,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
            activation=hparams.activation,
        )
        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=hparams.encoder_num_layers,
        )

    def _build_decoder(self, hparams, in_channels):
        self._project_input_tgt = ops.init_layer(
            nn.Linear(in_channels + self.style_channels, hparams.d_model), init_type="xavier_uniform"
        )
        # create the decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hparams.d_model,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            dropout=hparams.dropout,
            activation=hparams.activation,
        )
        self._transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=hparams.decoder_num_layers
        )

    def forward(self, inputs: Tensor, z_style: Optional[Tensor] = None, **kwargs) -> Tensor:

        x = inputs  # N, L, C
        if z_style is not None:
            assert z_style.ndim == 2 or (z_style.ndim == 3 and z_style.shape[1] == 1)
            if z_style.ndim == 2:
                z_style = z_style.unsqueeze(1)
            x = torch.cat((x, z_style.expand(-1, x.shape[1], -1)), dim=2)

        # project input
        src = self._project_input(x)
        # transpose input N,L,C -> L,N,C
        src = src.permute(1, 0, 2).contiguous()
        # src = self._pos_enc(src)
        # encode src as memory
        assert src.shape[0] == self.src_seq_frames + self.seq_pad
        assert src.shape[2] == self.hparams.d_model
        mem = self._transformer_encoder(src, mask=self.mask)

        # decoder
        tgt = self._project_input_tgt(x)
        tgt = tgt.permute(1, 0, 2).contiguous()
        # tgt = self._pos_dec(tgt)
        assert tgt.shape[0] == self.tgt_seq_frames + self.seq_pad
        assert tgt.shape[2] == self.hparams.d_model
        out = self._transformer_decoder(tgt=tgt, memory=mem, tgt_mask=self.mask, memory_mask=self.mask)
        out = out.permute(1, 0, 2)  # L,N,C -> N,L,C

        # remove the padding
        out = out[:, self.seq_pad :, :]
        return out
