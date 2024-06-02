import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import AutoPadding
from src.modules.layers import Conv1d, Linear

"""
[0] [1] [2] ... [n-1]

qry = Linear([1])
key = Linear([0]+[1] [1]+[1] [2]+[2])
val = [0] [1] [2]
"""


class Smoother(nn.Module):
    def __init__(self, in_channels, n_units, win_size):
        super().__init__()
        self._win_size = win_size
        self._W_qry = Linear(in_channels, n_units, bias=False)
        self._W_key = Linear(in_channels, n_units, bias=False)
        self.v = Linear(n_units, 1, bias=False)
        self.b = torch.nn.Parameter(torch.zeros((1, 1, 1, n_units)))

    def windowing(self, x):
        N, L, C = x.shape
        idx = torch.arange(0, L - self._win_size + 1, dtype=torch.long).unsqueeze(1).repeat(1, self._win_size)
        idx += torch.arange(0, self._win_size, dtype=torch.long).unsqueeze(0)
        n_w = len(idx)
        # * flatten
        idx = idx.view(n_w * self._win_size).to(x.device)
        x_w_flatten = torch.index_select(x, dim=1, index=idx)  # N, n_w*wsize, C
        x_w = x_w_flatten.view(N, n_w, self._win_size, C)  # N, n_w, win_size, C
        return x_w

    def forward(self, x):
        assert x.ndim == 3
        x_win = self.windowing(x)
        x_ctr = x[:, self._win_size // 2 : self._win_size // 2 + x_win.shape[1]].unsqueeze(2)
        key = self._W_key(x_win)
        qry = self._W_qry(x_ctr)
        score = self.v(torch.tanh(qry + key + self.b))
        # print(x_win.shape, x_ctr.shape, score.shape)
        weight = F.softmax(score, dim=2)
        out = (x_win * weight).sum(dim=2)
        return out
