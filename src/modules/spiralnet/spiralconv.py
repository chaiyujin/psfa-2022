import torch
import torch.nn as nn


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.register_buffer("indices", indices)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        assert x.shape[1] == n_nodes, "[SpiralConv]: Input has {} nodes, but should be {}".format(x.shape[1], n_nodes)
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError("x.dim() is expected to be 2 or 3, but received {}".format(x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return "{}({}, {}, seq_length={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.seq_length,
        )
