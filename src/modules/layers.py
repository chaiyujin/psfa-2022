import math

import torch
import torch.nn as nn

from src.engine.ops import init_layer, parse_activation, parse_norm
from src.engine.seq_ops import fold, unfold


def Conv2d(*args, **kwargs):
    init_kwargs = kwargs.get("init_kwargs", {})
    kwargs.pop("init_kwargs", None)
    m = nn.Conv2d(*args, **kwargs)
    return init_layer(m, **init_kwargs)


def Conv1d(*args, **kwargs):
    init_kwargs = kwargs.get("init_kwargs", {})
    kwargs.pop("init_kwargs", None)
    m = nn.Conv1d(*args, **kwargs)
    return init_layer(m, **init_kwargs)


def Linear(*args, **kwargs):
    init_kwargs = kwargs.get("init_kwargs", {})
    kwargs.pop("init_kwargs", None)
    m = nn.Linear(*args, **kwargs)
    return init_layer(m, **init_kwargs)


def LSTM(*args, **kwargs):
    init_kwargs = kwargs.get("init_kwargs", {})
    init_kwargs["init_type"] = "orthogonal"  # must be 'orthogonal'
    kwargs.pop("init_kwargs", None)
    m = nn.LSTM(*args, **kwargs)
    return init_layer(m, **init_kwargs)


class PositionalEncodingLNC(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(1, 0))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        norm_method="none",
        activation="lrelu0.2",
        last_activation=None,
        init_kwargs={},
    ):
        super().__init__()
        if isinstance(out_channels, int):
            out_channels = [out_channels]

        inps = (in_channels,) + tuple(out_channels[:-1])
        outs = tuple(out_channels)
        if isinstance(activation, str):
            acts = [activation for _ in outs]
            if last_activation is not None:
                acts[-1] = last_activation
        else:
            acts = list(activation)
            assert last_activation is None, "You give activation by list, should not set 'last_activation'!"
            assert len(acts) == len(inps), "Given {} activations, but {} layers".format(len(acts), len(inps))

        # build layers
        layers = []
        for i_layer, (inp, out, act) in enumerate(zip(inps, outs, acts)):
            layers.append(Linear(inp, out, bias=bias, init_kwargs=init_kwargs))
            if i_layer + 1 < len(inps) and norm_method != "none":
                layers.append(parse_norm(norm_method, out))
            layers.append(parse_activation(act))

        self._main = nn.ModuleList(layers)
        self._norm = norm_method
        self.in_channels = in_channels
        self.out_channels = out_channels[-1]

    def forward(self, x):
        # check input shape
        assert (
            x.ndim in [2, 3] and x.shape[-1] == self.in_channels
        ), "Input has wrong shape: {0}, should be 2 (N, {1}) or 3 (N, L, {1}) dim!".format(x.shape, self.in_channels)

        if x.ndim == 2 or self._norm == "none":
            for layer in self._main:
                x = layer(x)
        else:
            for layer in self._main:
                lname = layer.__class__.__name__
                if lname.find("InstanceNorm") >= 0:
                    x = x.permute(0, 2, 1)  # N, C, L
                    x = layer(x)
                    x = x.permute(0, 2, 1)  # N, L, C
                elif lname.find("Linear") >= 0 or lname.find("BatchNorm1d") >= 0:
                    x, frm = fold(x)  # NL, C
                    x = layer(x)
                    x = unfold(x, frm)  # N, L, C
                else:
                    x = layer(x)
        return x
