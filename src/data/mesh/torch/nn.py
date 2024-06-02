import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


def pool_bvc(x: Tensor, mat: Tensor):
    assert x.ndim == 3 and mat.ndim == 2
    assert mat.is_sparse, "Only support torch SparseTensor!"
    dim = 1  # dim 'v'

    assert x.shape[dim] == mat.shape[1]
    mat = mat.to(x.device)
    row, col = mat._indices()
    val = mat._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * val
    out = scatter_add(out, row, dim, dim_size=mat.size(0))
    return out


def message_passing_bvc(message_passing: MessagePassing, x: Tensor, e: Tensor, *args, **kwargs):
    # check input shape
    assert x.ndim == 3, "'x' shape must be (batch_size, n_verts, n_channels), but {}".format(x.shape)
    assert (
        e.ndim in [2, 3] and e.shape[-2] == 2
    ), "'e' shape must be (batch_size, 2, n_edges) or (2, n_edges), but {}".format(e.shape)
    bsz, nv, c = x.shape
    # expand edges if necessary
    if e.ndim == 2:
        e = e.unsqueeze(0).expand(bsz, -1, -1)
    # make sure same batch size and same device
    assert x.shape[0] == e.shape[0], "'x', 'e' have different batch size {} != {}".format(x.shape[0], e.shape[0])
    e = e.to(x.device)
    # prepare into [N, C] and [2, E] for message passing
    packed_x = x.view(bsz * nv, c)
    packed_e = pack_edges(e, nv)
    # message passing
    packed_y = message_passing(packed_x, edge_index=packed_e)
    # reshape into batched version
    y = packed_y.view(bsz, nv, -1)
    return y


def pack_edges(edges, num_verts):
    assert edges.shape[1] == 2
    e = edges.permute(1, 0, 2).contiguous()
    e = e + (torch.arange(e.shape[1], device=e.device) * num_verts)[None, :, None]
    return e.view(2, -1)


def unpack_edges(packed_edges, bsz, num_verts):
    assert packed_edges.shape[0] == 2
    e = packed_edges.view(2, bsz, -1)
    e = e - (torch.arange(e.shape[1], device=e.device) * num_verts)[None, :, None]
    return e.permute(1, 0, 2).contiguous()
