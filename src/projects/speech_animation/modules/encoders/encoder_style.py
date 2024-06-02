from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.ops import to_onehot
from src.engine.seq_ops import fold, unfold
from src.modules.layers import MLP


def build_encoder_style(hparams) -> Tuple[nn.Module, int]:
    # check if using
    name = hparams.using
    # build
    _build_dict = dict(
        embed=_SpeakerEmbedding,
        onehot=_SpeakerOnehot,
        stl=_StyleTokenLayer,
        sve=_StyleVariationalEncoder,
    )
    assert name in _build_dict, f"style encoder is using unknown module '{name}'"
    m = _build_dict[name](hparams)  # type: ignore
    return m, m.latent_size


class _SpeakerEmbedding(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        opt = hparams.embed
        self._embed = nn.Embedding(hparams.n_speakers, opt.embedding_size)
        self._embed.weight.data.normal_(0, opt.init_std)
        self.n_speakers = hparams.n_speakers
        self.latent_size = opt.embedding_size

    def forward(self, style_id, **kwargs):
        if kwargs.get("style_comb") is not None:
            x = kwargs["style_comb"]
            assert x.ndim == 2
            return torch.matmul(x, self._embed.weight)
        assert style_id.ndim == 1
        embeddings = self._embed(style_id)
        return embeddings


class _SpeakerOnehot(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_speakers = hparams.n_speakers
        self.latent_size = self.n_speakers

    def forward(self, style_id, **kwargs):
        if kwargs.get("style_comb") is not None:
            return kwargs["style_comb"]
        assert style_id.ndim == 1
        return to_onehot(style_id, self.n_speakers).float()


class _StyleTokenLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_q = config.stl.d_model // 2
        d_k = config.stl.d_model // config.stl.n_heads
        self.embed = nn.Parameter(torch.FloatTensor(config.stl.n_tokens, config.stl.d_model // config.stl.n_heads))
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, n_units=config.stl.d_model, n_heads=config.stl.n_heads
        )
        self.latent_size = config.stl.d_model
        self.n_tokens = config.stl.n_tokens

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, style_ref, **kwargs):
        if kwargs.get("style_comb") is not None:
            return self.combine(kwargs["style_comb"])

        frm = None
        if style_ref.ndim == 3:
            style_ref, frm = fold(style_ref)

        assert style_ref.ndim == 2
        N = style_ref.size(0)
        query = style_ref.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // n_heads]
        style_embed = self.attention(query, keys, average=kwargs.get("style_ntrl")).squeeze(1)

        if frm is not None:
            style_embed = unfold(style_embed, frm)
        return style_embed

    def combine(self, style_comb):
        if style_comb.dtype == torch.long:
            style_comb = to_onehot(style_comb, self.n_tokens)

        frm = None
        if style_comb.ndim == 3:
            style_comb, frm = fold(style_comb)

        N = style_comb.shape[0]
        assert style_comb.shape[1] == self.n_tokens
        assert style_comb.ndim == 2
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // n_heads]
        vals = self.attention.W_val(keys)  # [N, token_num, n_units]
        style_embed = (style_comb.unsqueeze(-1) * vals).sum(dim=1)

        if frm is not None:
            style_embed = unfold(style_embed, frm)
        return style_embed


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, n_units]
    """

    def __init__(self, query_dim, key_dim, n_units, n_heads):

        super().__init__()
        self.n_units = n_units
        self.n_heads = n_heads
        self.key_dim = key_dim

        self.W_qry = nn.Linear(in_features=query_dim, out_features=n_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=n_units, bias=False)
        self.W_val = nn.Linear(in_features=key_dim, out_features=n_units, bias=False)

    def forward(self, query, key, average=False):
        qrys = self.W_qry(query)  # [N, T_q, n_units]
        keys = self.W_key(key)  # [N, T_k, n_units]
        vals = self.W_val(key)

        split_size = self.n_units // self.n_heads
        qrys = torch.stack(torch.split(qrys, split_size, dim=2), dim=0)  # [h, N, T_q, n_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, n_units/h]
        vals = torch.stack(torch.split(vals, split_size, dim=2), dim=0)  # [h, N, T_k, n_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(qrys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim**0.5)
        scores = F.softmax(scores, dim=3)
        if average:
            scores = scores.detach()
            scores.data.fill_(1.0 / scores.shape[3])

        # out = score * V
        out = torch.matmul(scores, vals)  # [h, N, T_q, n_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, n_units]

        return out


class _StyleVariationalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._in_chs = config.sve.in_channels
        self.latent_size = config.sve.z_size
        self.mlp = MLP(
            self._in_chs,
            [256, 256, 256, self.latent_size * 2],
            norm_method=config.sve.norm_method,
            activation=config.sve.activation,
            last_activation="identity",
        )

    def forward(self, style_ref, **kwargs):
        frm = None
        if style_ref.ndim == 3:
            style_ref, frm = fold(style_ref)

        assert style_ref.ndim == 2
        x = self.mlp(style_ref)
        # get gaussian parameters
        loc, logs = x.chunk(chunks=2, dim=1)
        logs = torch.clamp(logs, min=-7.0)
        # sample
        if self.training:
            dist = torch.distributions.Normal(loc=torch.zeros_like(loc), scale=torch.ones_like(logs))
            style_embed = dist.sample() * torch.exp(logs) + loc
        else:
            # print("hahaha")
            style_embed = loc

        if frm is not None:
            style_embed = unfold(style_embed, frm)
            loc = unfold(loc, frm)
            logs = unfold(logs, frm)
        return (style_embed, (loc, logs))
