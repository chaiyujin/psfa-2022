import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.seq_ops import fold, unfold
from src.modules.layers import MLP

from ..modules.decoders import build_decoder_verts
from ..modules.encoders import build_encoder_audio, build_encoder_refxy, build_encoder_style


class AnimNetMixer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # checking
        assert config.company_audio == "win"
        self.config = config
        self.using_audio_feature = config.using_audio_feature
        self.src_seq_pads = tuple(config.get("src_seq_pads", [0, 0]))

        # * Audio encoder
        self.encoder_audio, ch_audio = build_encoder_audio(config.encoder.audio)

        # * Refer encoder
        self.encoder_reflw, ch_reflw = build_encoder_refxy(config.encoder.refxy_lower)
        self.encoder_refup, ch_refup = build_encoder_refxy(config.encoder.refxy_upper)

        # # * Z's style embedding
        # self.encoder_sty_z, ch_sty_z = build_encoder_style(config.encoder.style_z)

        # * Referenence lower part decomposition
        ch_out = ch_audio + config.encoder.ch_style_xy
        self.decomp = MLP(ch_reflw, [512, 512, 512, ch_out], last_activation="identity")
        self.ch_styxy = config.encoder.ch_style_xy

        # * Verts decoder
        ch_in = ch_audio + self.ch_styxy + ch_refup  # + ch_sty_z
        self.fusion = MLP(ch_in, 128, activation="identity")
        self.decoder_verts = build_decoder_verts(config.decoder.verts, 128)

        # * Viseme classifier
        self.classifier = MLP(ch_audio, [512, 512, config.n_visemes], last_activation="identity")

        self.ctt_method = config.encoder.ctt_method
        assert self.ctt_method in ["gate", "dot"]

        self.ctt_embed = nn.Parameter(torch.zeros((config.n_visemes, ch_audio)))
        self.ctt_embed.data.normal_(mean=0, std=1.0)
        if self.ctt_method == "gate":
            self.gates = MultiHeadGates(ch_audio, ch_audio, ch_audio, 256, 4)

    def remove_padding(self, x):
        if sum(self.src_seq_pads) > 0:
            padl, padr = self.src_seq_pads
            x = x[:, padl : x.shape[1] - padr]
        return x

    def decode(self, z_seq, idle_verts):
        # fold
        z, frm = fold(z_seq)
        idle_verts, _ = fold(idle_verts.expand(-1, frm, -1, -1))
        # decode
        out_verts, _ = self.decoder_verts(z, idle_verts)
        # unfold
        out_verts = unfold(out_verts, frm)
        return out_verts

    def _cate_content(self, latent_ctt):
        N = latent_ctt.shape[0]
        key = torch.tanh(self.ctt_embed).unsqueeze(0).expand(N, -1, -1)
        qry = latent_ctt.unsqueeze(1)
        z_d, prob_fine = self.gates(qry, key)
        z_d, prob_fine = z_d.squeeze(1), prob_fine.squeeze(1)
        return z_d

    def forward(self, idle_verts, audio_dict, offsets_refer, style_id, **kwargs):
        # * Idle verts -> (N,1,V,3)
        if idle_verts is not None and idle_verts.ndim == 3:
            idle_verts = idle_verts.unsqueeze(1)

        # * Get audio feature
        assert audio_dict is not None
        self.using_audio_feature = self.config.encoder.audio.feature
        in_audio = audio_dict[self.using_audio_feature]

        # * Encode audio and lower face reference
        in_audio, frm = fold(in_audio)
        in_refer, frm = fold(offsets_refer)

        # # style id -> 'z' component style
        # in_styid, _ = self._expand(style_id, frm, tar_dim=2)
        # latent_sty_z = self.encoder_sty_z(in_styid)

        # audio
        latent_ctt_audio = self.encoder_audio(in_audio)
        # refer lower, upper
        latent_reflw = self.encoder_reflw(in_refer)
        latent_refup = self.encoder_refup(in_refer)

        # * Decompose latent_reflw -> latent_ctt_refer, latent_sty_xy
        y = self.decomp(latent_reflw)
        latent_sty_reflw = y[..., : self.ch_styxy]
        latent_ctt_refer = y[..., self.ch_styxy :]

        if self.ctt_method == "gate":
            # print("yes!!")
            latent_ctt_audio = self._cate_content(latent_ctt_audio)
            latent_ctt_refer = self._cate_content(latent_ctt_refer)

        # * Classify
        logits_ctt_audio = self.classifier(latent_ctt_audio)
        logits_ctt_refer = self.classifier(latent_ctt_refer)

        if self.ctt_method == "dot":
            # print("no!!")
            mat = torch.tanh(self.ctt_embed)
            latent_ctt_audio = torch.matmul(F.softmax(logits_ctt_audio, dim=-1), mat)
            latent_ctt_refer = torch.matmul(F.softmax(logits_ctt_refer, dim=-1), mat)

        # * Decode output
        def _fuse(z_ctt):
            # z = torch.cat((z_ctt, latent_sty_reflw, latent_sty_z), dim=-1)
            z = torch.cat((z_ctt, latent_sty_reflw, latent_refup), dim=-1)
            return self.fusion(z)

        def _fuse_no_style(z_ctt):
            z = torch.cat((z_ctt, torch.zeros_like(latent_sty_reflw), torch.zeros_like(latent_refup)), dim=-1)
            return self.fusion(z)

        from_ctt_a = _fuse(latent_ctt_audio)
        from_ctt_r = _fuse(latent_ctt_refer)
        from_ctt_a_nosty = _fuse_no_style(latent_ctt_audio)
        from_ctt_r_nosty = _fuse_no_style(latent_ctt_refer)

        verts_for_audio = self.decode(unfold(from_ctt_a, frm), idle_verts)
        verts_for_reflw = self.decode(unfold(from_ctt_r, frm), idle_verts)
        verts_for_audio_nostyle = self.decode(unfold(from_ctt_a_nosty, frm), idle_verts)
        verts_for_reflw_nostyle = self.decode(unfold(from_ctt_r_nosty, frm), idle_verts)

        return dict(
            out_verts_audio=verts_for_audio,
            out_verts_refer=verts_for_reflw,
            out_verts_audio_nostyle=verts_for_audio_nostyle,
            out_verts_refer_nostyle=verts_for_reflw_nostyle,
            logits_ctt_audio=unfold(logits_ctt_audio, frm),
            logits_ctt_refer=unfold(logits_ctt_refer, frm),
            latent_ctt_audio=unfold(latent_ctt_audio, frm),
            latent_ctt_refer=unfold(latent_ctt_refer, frm),
        )

    def _expand(self, x, frm, tar_dim):
        assert x.ndim in [tar_dim - 1, tar_dim]
        if x.ndim == tar_dim - 1:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            shape = [-1 for _ in range(x.ndim)]
            shape[1] = frm
            x = x.expand(*shape)
        return fold(x)


class MultiHeadGates(nn.Module):
    def __init__(self, qry_dim, key_dim, val_dim, n_units, n_heads):

        super().__init__()
        self.n_units = n_units
        self.val_dim = val_dim
        self.n_heads = n_heads

        self.proj_qry = MLP(qry_dim, [n_units, n_units], bias=False, activation="lrelu0.2", last_activation="identity")
        self.proj_key = MLP(key_dim, [n_units, n_units], bias=False, activation="lrelu0.2", last_activation="identity")
        self.proj_val = MLP(key_dim, [n_units, val_dim], bias=False, activation="lrelu0.2", last_activation="identity")

    def forward(self, in_qry, in_key, dropout_mask=None):
        assert in_qry.ndim == 3 and in_key.ndim == 3
        assert in_qry.shape[0] == in_key.shape[0]
        # * project into same channels
        qrys = self.proj_qry(in_qry)  # [N, T_q, n_units]
        keys = self.proj_key(in_key)  # [N, T_k, n_units]
        vals = self.proj_val(in_key)  # [N, T_k, n_units]

        # * split into multi-heads
        qrys = torch.stack(torch.split(qrys, self.n_units // self.n_heads, dim=2), dim=0)  # [h, N, T_q, n_units/h]
        keys = torch.stack(torch.split(keys, self.n_units // self.n_heads, dim=2), dim=0)  # [h, N, T_k, n_units/h]
        vals = torch.stack(torch.split(vals, self.val_dim // self.n_heads, dim=2), dim=0)  # [h, N, T_k, n_units/h]

        # * Dot Production Attention: softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(qrys, keys.permute(0, 1, 3, 2))  # [h, N, T_q, T_k]
        scores = scores / (keys.shape[3] ** 0.5)
        scores = F.softmax(scores, dim=3)
        # ! cannot use sigmoid here, it's highly possible that sigmoid outputs all zeros

        # * out = score * V
        out = torch.matmul(scores, vals)  # [h, N, T_q, n_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, n_units]
        scores = torch.cat(torch.split(scores, 1, dim=0), dim=3).squeeze(0)

        return out, scores
