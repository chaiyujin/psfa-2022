import functools
import hashlib
import os
import pickle
from copy import copy
from typing import List, Optional, Tuple, Union

import frozendict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix
from torch import Tensor
from torch_geometric.nn import ChebConv as _ChebConv
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
from torch_scatter import scatter_add

from assets import ASSETS_ROOT
from src.data.mesh.numpy import Mesh, generate_sampling_information
from src.data.mesh.torch import nn as meshnn
from src.engine.logging import get_logger
from src.engine.ops import is_identity_activation, parse_activation, sparse_from_csc
from src.modules.spiralnet import SpiralConv, get_spiral_indices

log = get_logger(__name__)


class ChebConv(_ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization="sym", bias=True, **kwargs):
        super().__init__(in_channels, out_channels, K, normalization, bias, **kwargs)

    def pre_norm(self, edge_index, num_nodes):
        assert num_nodes <= 5023, "For FLAME mesh, should only <= 5023 nodes"

        self._one_nv = num_nodes
        self._one_edge, self._one_norm = ChebConv.norm(edge_index.long(), num_nodes, dtype=torch.float32)

    @staticmethod
    def norm(edge_index, num_nodes: Optional[int], edge_weight: OptTensor = None, dtype: Optional[int] = None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x):
        bsz, nv, _ = x.shape
        assert nv == self._one_nv

        # pack verts
        x = x.view(bsz * nv, -1)

        # pack edge index and norm
        edge_index = self._one_edge.to(x.device).unsqueeze(0).expand(bsz, -1, -1)
        edge_index = meshnn.pack_edges(edge_index, nv)
        norm = self._one_norm.to(x.device).unsqueeze(0).repeat(bsz, 1).view(edge_index.shape[1])

        """ ChebConv Foward in shape: x [N,C], edge_index [2,E], norm [E] """
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + torch.matmul(Tx_1, self.weight[1])  # type: ignore

        for k in range(2, self.weight.size(0)):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0  # type: ignore
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias
        """ ChebConv is done """

        # unpack and return
        out = out.view(bsz, nv, -1)
        return out


def _to_tensor(t):
    if t is not None and not torch.is_tensor(t):
        if isinstance(t, csc_matrix):
            return sparse_from_csc(t)
        elif isinstance(t, np.ndarray):
            return torch.from_numpy(t)
    return t


class _Block(nn.Module):
    def __init__(
        self,
        block_type: str,
        conv: nn.Module,
        sampling_matrix: Tensor,
        edge_index: Optional[Tensor] = None,
        activation: Union[str, nn.Module] = "elu",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert block_type.lower() in ["encoder", "decoder"]
        self._type = block_type.lower()

        # conv layer
        self.conv = conv
        # sampling matrix
        self.sampling_matrix = _to_tensor(sampling_matrix)
        # edge index
        self.register_buffer("edge_index", _to_tensor(edge_index))
        # activation
        self.activation = parse_activation(activation) if isinstance(activation, str) else activation
        # batch norm
        self.batch_norm = nn.BatchNorm1d(self.conv.out_channels) if batch_norm else None
        # dropout
        self.dropout = dropout

        # SPECIAL: custom ChebConv
        if isinstance(conv, ChebConv):
            conv.pre_norm(self.edge_index, self.n_verts_in if self._type == "encoder" else self.n_verts_out)
        # init parameters
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.conv.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = getattr(self, f"_run_as_{self._type}")(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _conv_and_post(self, x):
        # x in (batch_size, vert, feature) shape, forward by tool
        if isinstance(self.conv, (GCNConv)):
            assert self.edge_index is not None
            x = meshnn.message_passing_bvc(self.conv, x, self.edge_index)
        elif isinstance(self.conv, (ChebConv, SpiralConv)):
            x = self.conv(x)
        else:
            raise NotImplementedError("Unknown conv: {}".format(type(self.conv)))
        # batch_norm
        if self.batch_norm is not None:
            x = self.batch_norm(x.view(-1, x.shape[-1])).view(*x.shape)
        # activation
        x = self.activation(x)
        return x

    def _run_as_encoder(self, x):
        x = self._conv_and_post(x)
        if self.sampling_matrix is not None:
            x = meshnn.pool_bvc(x, self.sampling_matrix)
        return x

    def _run_as_decoder(self, x):
        if self.sampling_matrix is not None:
            x = meshnn.pool_bvc(x, self.sampling_matrix)
        x = self._conv_and_post(x)
        return x

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def n_verts_in(self):
        return self.sampling_matrix.shape[1]

    @property
    def n_verts_out(self):
        return self.sampling_matrix.shape[0]

    def __repr__(self):
        if self._type == "encoder":
            return "MeshConvBlock - {} (\n  conv: {},\n  edge_idx: {}\n  batch_norm: {}\n  activation: {}\n  sampling: {}\n)".format(
                self._type,
                self.conv,
                self.edge_index.shape,
                self.batch_norm,
                self.activation,
                "{}->{}".format(self.n_verts_in, self.n_verts_out),
            )
        else:
            return "MeshConvBlock - {} (\n  sampling: {}\n  conv: {},\n  edge_idx: {}\n  batch_norm: {}\n  activation: {}\n)".format(
                self._type,
                "{}->{}".format(self.n_verts_in, self.n_verts_out),
                self.conv,
                self.edge_index.shape,
                self.batch_norm,
                self.activation,
            )


class MeshConvBlocks(nn.Module):

    __valid_conv_types__ = ["SpiralConv", "GCNConv", "ChebConv"]

    def __init__(
        self,
        block_type: str,
        conv_type: str,
        in_channels: int,
        hidden_channels: Tuple[int, ...],
        activation: str = "elu",
        batch_norm: bool = False,
        last_activation: Optional[str] = None,
        edge_indices: Optional[Tuple[Tensor, ...]] = None,
        spiral_indices: Optional[Tuple[Tensor, ...]] = None,
        sampling_matrices: Optional[Tuple[Tensor, ...]] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        block_type = block_type.lower()
        assert block_type in ["encoder", "decoder"]
        # * get layer args
        inp_list = [in_channels] + copy(list(hidden_channels[:-1]))
        out_list = copy(list(hidden_channels))
        act_list = [activation for _ in inp_list]
        if last_activation is not None:
            act_list[-1] = last_activation

        # ! check
        self._check_args(conv_type, len(inp_list), edge_indices, spiral_indices, sampling_matrices, **kwargs)

        # * build layers
        self.blocks = nn.ModuleList()
        for i_block in range(len(inp_list)):
            conv = self._build_conv(conv_type, i_block, inp_list, out_list, spiral_indices=spiral_indices, **kwargs)
            actv = act_list[i_block]
            edge = edge_indices[i_block] if edge_indices is not None else None
            mtrx = sampling_matrices[i_block] if sampling_matrices is not None else None
            block = _Block(
                block_type,
                conv,
                sampling_matrix=mtrx,
                edge_index=edge,
                activation=actv,
                batch_norm=batch_norm if not is_identity_activation(actv) else False,
                dropout=dropout,
            )
            self.blocks.append(block)

        self.reset_parameters()

    def reset_parameters(self):
        # each block reset themselves
        pass

    def forward(self, x):
        # print('input', x.shape)
        for block in self.blocks:
            x = block(x)
            # print('~ ', x.shape)
        return x

    @property
    def in_channels(self):
        return self.blocks[0].in_channels

    @property
    def out_channels(self):
        return self.blocks[-1].out_channels

    @classmethod
    def _check_args(cls, conv_type, n_layers, edge_indices, spiral_indices, sampling_matrices, **kwargs):
        # check
        assert conv_type in cls.__valid_conv_types__
        if conv_type == "SpiralConv":
            assert spiral_indices is not None
            assert len(spiral_indices) == n_layers, "{} layers, but {} spiral_indices are given!".format(
                n_layers, len(spiral_indices)
            )
        else:
            assert edge_indices is not None
            assert len(edge_indices) == n_layers, "{} layers, but {} edge_indices are given!".format(
                n_layers, len(edge_indices)
            )
            if conv_type == "ChebConv":
                assert kwargs.get("K") is not None

        if sampling_matrices is not None:
            assert len(sampling_matrices) == n_layers, "{} layers, but {} sampling_matrices are given!".format(
                n_layers, len(sampling_matrices)
            )

    @classmethod
    def _build_conv(cls, conv_type, idx, inp_list, out_list, spiral_indices, **kwargs):
        inp, out = inp_list[idx], out_list[idx]
        if conv_type == "SpiralConv":
            return SpiralConv(inp, out, indices=spiral_indices[idx])
        elif conv_type == "GCNConv":
            return GCNConv(inp, out)
        elif conv_type == "ChebConv":
            K = kwargs["K"] if isinstance(kwargs["K"], int) else kwargs["K"][idx]
            return ChebConv(inp, out, K=K)
        else:
            raise NotImplementedError("unknown conv_type: {}".format(conv_type))


def freezeargs(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    def freeze(v):
        if isinstance(v, dict):
            return frozendict(v)
        elif isinstance(v, list):
            return tuple(v)
        return v

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([freeze(arg) for arg in args])
        kwargs = {k: freeze(kwargs[k]) for k in sorted(list(kwargs.keys()))}
        return func(*args, **kwargs)

    return wrapped


@freezeargs
@functools.lru_cache(maxsize=1)
def preprocess_template(template_path, ds_factors, spiral_seq_lengths=None, spiral_dilations=None):
    # cache_key = (os.path.abspath(template_path), ds_factors, spiral_seq_lengths, spiral_dilations)
    # cache_fname = hashlib.md5(str(cache_key).encode()).hexdigest() + ".pkg"
    # cache_fname = (
    #     os.path.basename(template_path) +
    #     "-ds_" + "_".join(str(x) for x in ds_factors) +
    #     "-spseq_" + "_".join(str(x) for x in spiral_seq_lengths) +
    #     "-spdil_" + "_".join(str(x) for x in spiral_dilations)
    # )
    # cache_fpath = os.path.join(ASSETS_ROOT, ".mesh_tmpl_info", cache_fname + ".pkg")
    # if os.path.exists(cache_fpath):
    #     # print('Load cached template information from: {}'.format(cache_fpath))
    #     with open(cache_fpath, "rb") as fp:
    #         info = pickle.load(fp)
    #     return info

    mesh = Mesh.LoadFile(os.path.expanduser(template_path))
    info = generate_sampling_information(mesh.v, mesh.f, ds_factors)
    # (optional) spiral indices
    if spiral_seq_lengths is not None:
        if spiral_dilations is None:
            spiral_dilations = [1 for _ in spiral_seq_lengths]
        spiral_indices = get_spiral_indices(info["M"], spiral_seq_lengths, spiral_dilations)
    else:
        spiral_indices = None
    info = dict(
        M=info["M"],
        D=info["D"],
        U=info["U"],
        edge_indices=info["E"],
        spiral_indices=spiral_indices,
    )

    # for m in info["D"]:
    #     print("downsample", m.shape)
    # dump_info = dict(
    #     M=list(),
    #     D=info["D"],
    #     U=info["U"],
    #     edge_indices=info["edge_indices"],
    #     spiral_indices=spiral_indices,
    # )
    # for m in info["M"]:
    #     dump_info["M"].append(dict(v=m.v, f=m.f))
    # os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
    # with open(cache_fpath, "wb") as fp:
    #     pickle.dump(dump_info, fp)
    # quit(1)

    return info


def build_mesh_conv_blocks(
    template_path: str,
    block_type: str,
    conv_type: str,
    in_channels: int,
    hidden_channels: Tuple[int, ...],
    activation: str = "elu",
    batch_norm: bool = False,
    last_activation: Optional[str] = None,
    ds_factors: Tuple[int, ...] = (4, 4, 4),
    spiral_seq_lengths: Optional[Union[int, Tuple[int, ...]]] = None,
    spiral_dilations: Optional[Union[int, Tuple[int, ...]]] = None,
    dropout: float = 0.0,
    **kwargs,
):
    # auto int -> tuple
    if spiral_seq_lengths is not None:
        if isinstance(spiral_seq_lengths, int):
            spiral_seq_lengths = tuple([spiral_seq_lengths for _ in ds_factors])
        if spiral_dilations is None:
            spiral_dilations = 1
        if isinstance(spiral_dilations, int):
            spiral_dilations = tuple([spiral_dilations for _ in ds_factors])

    # fetch the indices and sampling things
    tmpl_info = preprocess_template(
        template_path=template_path,
        ds_factors=ds_factors,
        spiral_seq_lengths=spiral_seq_lengths,
        spiral_dilations=spiral_dilations,
    )
    if block_type.lower() == "encoder":
        edge_indices = tmpl_info["edge_indices"][:-1]
        spiral_indices = tmpl_info["spiral_indices"]
        sampling_matrices = tmpl_info["D"]
    elif block_type.lower() == "decoder":
        edge_indices = tuple(reversed(tmpl_info["edge_indices"][:-1]))
        spiral_indices = tuple(reversed(tmpl_info["spiral_indices"]))
        sampling_matrices = tuple(reversed(tmpl_info["U"]))
    else:
        raise ValueError("unknown block_type: {}".format(block_type))

    return MeshConvBlocks(
        block_type=block_type,
        conv_type=conv_type,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        batch_norm=batch_norm,
        last_activation=last_activation,
        edge_indices=edge_indices,
        spiral_indices=spiral_indices,
        sampling_matrices=sampling_matrices,
        dropout=dropout,
        **kwargs,
    )
