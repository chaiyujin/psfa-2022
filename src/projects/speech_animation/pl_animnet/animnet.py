import torch
import torch.nn as nn

from src.engine.logging import get_logger
from src.engine.seq_ops import fold, unfold, unfold_dict

from ..modules.decoders import build_decoder_verts
from ..modules.encoders import build_encoder_audio, build_encoder_style
from ..modules.sequence import build_sequential

log = get_logger("AnimNet")


class AnimNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # * Audio information
        assert config.company_audio == "win"
        self.using_audio_feature = config.using_audio_feature

        # * Audio encoder
        self.encoder_audio, ch_audio = build_encoder_audio(config.encoder.audio)

        # * Style encoder (Optional)
        if "style" in self.config.encoder.using:
            self.encoder_style, ch_style = build_encoder_style(config.encoder.style)
            log.info(f">>> Styles: {self.encoder_style.n_speakers}")
        else:
            self.encoder_style, ch_style = None, 0

        # * Sequential module
        self.seq_module = build_sequential(
            config.sequential,
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

    def forward(self, idle_verts, audio_dict, **kwargs):
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

        # * (Optional) encode style
        z_style = None
        if self.encoder_style is not None:
            z_style = self.encoder_style(kwargs.get("style_id"))

        # * Sequential, may fuse style
        z = self.seq_module(z_audio, z_style=z_style, **kwargs)

        # * Decode
        outputs = self.decode(z, idle_verts)
        return outputs
