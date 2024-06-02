import torch
import torch.nn as nn

from src.engine.logging import get_logger
from src.engine.ops import freeze
from src.engine.seq_ops import fold, unfold, unfold_dict
from src.modules.layers import MLP

from ..modules.decoders import build_decoder_verts
from ..modules.encoders import build_encoder_audio, build_encoder_offsets
from ..modules.seq_conv import SeqConv

log = get_logger("AnimNetDecmp")


class AnimNetDecmp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Audio information
        assert config.company_audio == "win"
        self.using_audio_feature = config.using_audio_feature
        self.pads = config.src_seq_pads

        # Audio encoders
        self.enc_ctt_aud, ch_ctt_aud = build_encoder_audio(config.encoder.audio)
        self.enc_upp_aud, ch_upp_aud = build_encoder_audio(config.encoder.audio_upper)
        # Content encoders
        self.enc_ctt_ani, ch_ctt_ani = build_encoder_offsets(config.encoder.content)
        self.enc_upp_ani, ch_upp_ani = build_encoder_offsets(config.encoder.content_upper)
        # Check
        assert ch_ctt_aud == ch_ctt_ani, f"Audio z_ctt channels ({ch_ctt_aud}) != anime ({ch_ctt_ani})"
        assert ch_upp_aud == ch_upp_ani, f"Audio z_upp channels ({ch_upp_aud}) != anime ({ch_upp_ani})"

        # Style, Content encoder
        self.enc_sty, ch_sty = build_encoder_offsets(config.encoder.style)
        self.rnn_sty = nn.LSTM(ch_sty, ch_sty, batch_first=True, bidirectional=False)

        # Sequential module
        self.seq_module = SeqConv(config.sequential.conv, ch_ctt_ani + ch_upp_ani, ch_sty)

        # * Verts decoder
        self.decoder_verts = build_decoder_verts(config.decoder.verts, self.seq_module.out_channels)

    def decode(self, z_seq, idle_verts):
        # * Inputs: idle verts -> N1V3
        if idle_verts is not None and idle_verts.ndim == 3:
            idle_verts = idle_verts.unsqueeze(1)
        # fold
        z, frm = fold(z_seq)
        idle_verts, _ = fold(idle_verts.expand(-1, frm, -1, -1))
        # decode
        out_verts, code_dict = self.decoder_verts(z, idle_verts)
        # unfold
        out_verts = unfold(out_verts, frm)
        code_dict = unfold_dict(code_dict, frm)
        return dict(out_verts=out_verts, code_dict=code_dict)

    def encode_audio(self, audio_dict, **kwargs):
        assert audio_dict is not None
        x_aud, frm = fold(audio_dict[self.using_audio_feature])
        x_upp, frm = fold(audio_dict["for_upper"])  # HACK: hard-coded 'for_upper'

        z_ctt = unfold(self.enc_ctt_aud(x_aud, **kwargs), frm)
        z_upp = unfold(self.enc_upp_aud(x_upp, **kwargs), frm)

        return z_ctt, z_upp

    def decomp_offsets(self, y):
        y, frm = (y, None) if y.ndim == 3 else fold(y)

        z_ctt = unfold(self.enc_ctt_ani(y), frm)
        z_upp = unfold(self.enc_upp_ani(y), frm)
        z_sty = unfold(self.enc_sty(y), frm)

        # INFO: remove the padded before conclude style latent
        if sum(self.pads) > 0:
            s, e = self.pads[0], z_sty.shape[1] - self.pads[1]
            z_sty = z_sty[:, s:e]
        _, (h, _) = self.rnn_sty(z_sty)
        z_sty = h[0].unsqueeze(1)

        return z_ctt, z_upp, z_sty

    def remix(self, z_ctt, z_upp, z_sty, idle_verts):
        # Sequential, may fuse style
        z = torch.cat((z_ctt, z_upp), dim=-1)
        z = self.seq_module(z, z_style=z_sty)
        # Decode
        return self.decode(z, idle_verts)
