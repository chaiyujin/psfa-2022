import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.layers import MLP


class Bank(nn.Module):
    def __init__(self, in_channels, d_model, n_tokens, n_heads):
        super().__init__()
        self.d_q = in_channels
        self.d_k = d_model
        self.n_tokens = n_tokens
        self.embed = nn.Parameter(torch.FloatTensor(self.n_tokens, self.d_k))
        self.attention = MultiHeadSelfAttention(self.d_q, self.d_k, n_units=d_model, n_heads=n_heads)
        self.latent_size = d_model

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, z_ref, **kwargs):
        assert z_ref.ndim == 2 and z_ref.shape[1] == self.d_q
        N = z_ref.size(0)
        qry = z_ref.unsqueeze(1)  # [N, 1, d_q]
        key = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, d_k]
        out, scores = self.attention(qry, key)
        return out.squeeze(1), scores.squeeze(2)


class MultiHeadSelfAttention(nn.Module):
    """
    input:
        qry: [N, T_q, qry_dim]
        key: [N, T_k, key_dim]
    output:
        out: [N, T_q, n_units]
    """

    def __init__(self, qry_dim, key_dim, n_units, n_heads):

        super().__init__()
        self.n_units = n_units
        self.n_heads = n_heads

        self.proj_qry = MLP(qry_dim, n_units, bias=False)
        self.proj_key = MLP(key_dim, n_units, bias=False)
        self.proj_val = MLP(key_dim, n_units, bias=False)

    def forward(self, in_qry, in_key):
        assert in_qry.ndim == 3 and in_key.ndim == 3
        assert in_qry.shape[0] == in_key.shape[0]
        # * project into same channels
        qrys = self.proj_qry(in_qry)  # [N, T_q, n_units]
        keys = self.proj_key(in_key)  # [N, T_k, n_units]
        vals = self.proj_val(in_key)  # [N, T_k, n_units]

        # * split into multi-heads
        split_size = self.n_units // self.n_heads
        qrys = torch.stack(torch.split(qrys, split_size, dim=2), dim=0)  # [h, N, T_q, n_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, n_units/h]
        vals = torch.stack(torch.split(vals, split_size, dim=2), dim=0)  # [h, N, T_k, n_units/h]

        # * Dot Production Attention: softmax(QK^T / sqrt(d_k))V
        d_k = keys.shape[3]
        scores = torch.matmul(qrys, keys.permute(0, 1, 3, 2))  # [h, N, T_q, T_k]
        scores = F.softmax(scores / (d_k**0.5), dim=3)

        # * out = score * V
        out = torch.matmul(scores, vals)  # [h, N, T_q, n_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, n_units]
        scores = torch.cat(torch.split(scores, 1, dim=0), dim=3).squeeze(0)

        return out, scores
