import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers import MLP, Linear
from src.utils import ops


class StyleTokenLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_q = config.stl.d_model // 2
        d_k = config.stl.d_model // config.stl.n_heads
        self.embed = nn.Parameter(torch.FloatTensor(config.stl.n_tokens, d_k))
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, n_units=config.stl.d_model, n_heads=config.stl.n_heads
        )
        self.latent_size = config.stl.d_model
        self.n_tokens = config.stl.n_tokens

        self.proj_audio = MLP(
            64,
            [512, 512, 256, config.stl.n_tokens],
            norm_method="bn1",
            activation="lrelu0.2",
            last_activation="identity",
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, style_ref, **kwargs):
        if kwargs.get("style_comb") is not None:
            raise NotImplementedError()
            # return self.combine(kwargs["style_comb"])

        assert "z_audio" in kwargs
        mask = self.proj_audio(kwargs["z_audio"])

        assert style_ref.ndim == 2
        N = style_ref.size(0)
        query = style_ref.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // n_heads]
        style_embed = self.attention(query, keys, mask=mask, average=kwargs.get("style_ntrl")).squeeze(1)

        return style_embed, mask

    def combine(self, style_comb):
        if style_comb.dtype == torch.long:
            style_comb = ops.to_onehot(style_comb, self.n_tokens)

        N = style_comb.shape[0]
        assert style_comb.shape[1] == self.n_tokens
        assert style_comb.ndim == 2
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // n_heads]
        vals = self.attention.W_val(keys)  # [N, token_num, n_units]
        style_embed = (style_comb.unsqueeze(-1) * vals).sum(dim=1)

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

    def forward(self, query, key, mask, average=False):
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
        scores = scores * mask[None, :, None, :]  # ! TRY MASK
        scores = F.softmax(scores, dim=3)
        if average:
            scores = scores.detach()
            scores.data.fill_(1.0 / scores.shape[3])

        # out = score * V
        out = torch.matmul(scores, vals)  # [h, N, T_q, n_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, n_units]

        return out
