import torch
import torch.nn as nn

from src.engine.logging import get_logger
from src.engine.seq_ops import fold, unfold, unfold_dict

from ..modules.decoders import build_decoder_verts
from ..modules.encoders import build_encoder_audio, build_encoder_offsets
from ..modules.seq_conv import SeqConv

log = get_logger("AnimNetRefer")


class AnimNetRefer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # * Audio information
        assert config.company_audio == "win"
        self.using_audio_feature = config.using_audio_feature
        self.pads = config.src_seq_pads

        # * Audio encoder
        self.encoder_audio, ch_audio = build_encoder_audio(config.encoder.audio)

        # * Style, Content encoder
        self.enc_sty, ch_style = build_encoder_offsets(config.encoder.style)
        self.rnn_sty = nn.LSTM(ch_style, ch_style, batch_first=True, bidirectional=False)

        # * Sequential module
        self.seq_module = SeqConv(
            config.sequential.conv,
            ch_audio,
            ch_style,
            self.config.src_seq_frames,
            self.config.tgt_seq_frames,
            self.config.src_seq_pads,
        )

        # * Verts decoder
        self.decoder_verts = build_decoder_verts(config.decoder.verts, self.seq_module.out_channels)

    def decode(self, z_seq, idle_verts):
        # fold
        z, frm = fold(z_seq)
        idle_verts, _ = fold(idle_verts.expand(-1, frm, -1, -1))
        # decode
        out_verts, code_dict = self.decoder_verts(z, idle_verts)
        # unfold
        out_verts = unfold(out_verts, frm)
        code_dict = unfold_dict(code_dict, frm)
        return dict(out_verts=out_verts, code_dict=code_dict)

    def encode_style(self, y):
        y, frm = (y, None) if y.ndim == 3 else fold(y)
        z_sty = unfold(self.enc_sty(y), frm)
        # INFO: remove the padded before conclude style latent
        if sum(self.pads) > 0:
            s, e = self.pads[0], z_sty.shape[1] - self.pads[1]
            z_sty = z_sty[:, s:e]
        _, (h, _) = self.rnn_sty(z_sty)
        z_sty = h[0].unsqueeze(1)
        return z_sty

    def forward(self, idle_verts, audio_dict, refer_offsets, **kwargs):
        # * Inputs: idle verts -> N1V3
        if idle_verts is not None and idle_verts.ndim == 3:
            idle_verts = idle_verts.unsqueeze(1)

        # * Inputs: audio feature
        assert audio_dict is not None
        self.using_audio_feature = self.config.encoder.audio.feature
        in_audio = audio_dict[self.using_audio_feature]

        # * Encode audio
        in_audio, frm = fold(in_audio)
        z_audio = self.encoder_audio(in_audio, **kwargs)
        z_audio = unfold(z_audio, frm)

        # * encode style
        if "cache" in kwargs:
            if "z_style" not in kwargs["cache"]:
                assert refer_offsets is not None
                if self.training:
                    assert refer_offsets.shape[1] == frm
                z_style = self.encode_style(refer_offsets)
                kwargs["cache"]["z_style"] = z_style.detach()
                # print('get z_sty', z_style.shape)
            else:
                z_style = kwargs["cache"]["z_style"]
                # print('reuse z_sty')
        else:
            if self.training:
                assert refer_offsets.shape[1] == frm
            z_style = self.encode_style(refer_offsets)

        # * Sequential, may fuse style
        z = self.seq_module(z_audio, z_style=z_style)

        # * Decode
        outputs = self.decode(z, idle_verts)
        return outputs
