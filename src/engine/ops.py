import logging
import re
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from scipy.sparse import csc_matrix
from torch import Tensor

log = logging.getLogger("ENGINE")


def sparse_from_csc(spmat: csc_matrix) -> Tensor:
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data),
        torch.Size(spmat.tocoo().shape),
    )


def get_affine_matrix(center, angle, scale):
    a = scale * torch.cos(angle)
    b = scale * torch.sin(angle)
    m00 = a
    m01 = b
    m02 = (1.0 - a) * center[..., 0] - b * center[..., 1]
    m10 = -b
    m11 = a
    m12 = b * center[..., 0] + (1.0 - a) * center[..., 1]
    theta = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
        ],
        dim=-2,
    )
    return theta


# -------------------------------------------------------------------------------------------------------------------- #
#                                                   About parameters                                                   #
# -------------------------------------------------------------------------------------------------------------------- #

_handlers_auto_eval = dict()


def auto_eval(m: nn.Module):
    def _hook(m: nn.Module, *args, **kwargs):
        m.eval()

    hash_val = m.__hash__()
    if hash_val not in _handlers_auto_eval:
        # log.debug(f"Module {m.__class__.__name__} will auto eval!")
        handler = m.register_forward_pre_hook(_hook)
        _handlers_auto_eval[hash_val] = handler
    else:
        log.warning(f"Module '{m.__class__.__name__}' has already been auto_eval hooked.")


def remove_auto_eval(m: nn.Module):
    hash_val = m.__hash__()
    assert hash_val in _handlers_auto_eval
    _handlers_auto_eval[hash_val].remove()


def freeze(m: Optional[nn.Module]) -> Optional[nn.Module]:
    """Freeze parameters and auto eval() before forward."""
    if m is None:
        return m
    if hasattr(m, "_yk_frozen") and m._yk_frozen:
        log.warning(f"'{m.__class__.__name__}' is frozen already! Ignore redundant freeze() operation.")
        return m

    for param in m.parameters():
        param.requires_grad = False
    m.apply(auto_eval)

    m._yk_frozen = True
    return m


def unfreeze(m: Optional[nn.Module]) -> Optional[nn.Module]:
    """ " Reverse freeze() operation."""
    if m is None:
        return m
    if not (hasattr(m, "_yk_frozen") and m._yk_frozen):
        log.warning(f"'{m.__class__.__name__}' is not frozen! Ignore unfreeze() operation.")
        return m

    for param in m.parameters():
        param.requires_grad = True
    m.apply(remove_auto_eval)

    delattr(m, "_yk_frozen")
    return m


def count_parameters(graph: nn.Module) -> Tuple[int, str]:
    parameters = filter(lambda p: p.requires_grad, graph.parameters())
    n_count = int(np.sum([np.prod(p.size()) for p in parameters]))
    n_count_str = [int(n_count // np.power(1000, s) % 1000) for s in range(8) if n_count // np.power(1000, s) > 0]
    for i in range(len(n_count_str) - 1):
        n_count_str[i] = "{:03d}".format(n_count_str[i])
    n_count_str[-1] = str(n_count_str[-1])
    return n_count, ",".join(reversed(n_count_str))


def count_buffers(graph: nn.Module) -> Tuple[int, str]:
    buffers = list(graph.buffers()) + list(filter(lambda p: not p.requires_grad, graph.parameters()))
    n_count = int(np.sum([np.prod(p.size()) for p in buffers]))
    n_count_str = [int(n_count // np.power(1000, s) % 1000) for s in range(8) if n_count // np.power(1000, s) > 0]
    for i in range(len(n_count_str) - 1):
        n_count_str[i] = "{:03d}".format(n_count_str[i])
    n_count_str[-1] = str(n_count_str[-1])
    return n_count, ",".join(reversed(n_count_str))


# -------------------------------------------------------------------------------------------------------------------- #
#                                                  One-hot convertion                                                  #
# -------------------------------------------------------------------------------------------------------------------- #


def to_onehot(tensor: Tensor, n: int, fill_value: float = 1.0) -> Tensor:
    vec = tensor.new_zeros(tensor.size() + (n,))
    vec.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_value)
    return vec


# -------------------------------------------------------------------------------------------------------------------- #
#                                                   Tensor reduction                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


def _pos_and_sort_dims(dim: Iterable[int], max_ndim: int) -> List[int]:
    ret = [d if d >= 0 else (d + max_ndim) for d in dim]
    # assert len(ret) == len(set(ret)), "Duplicated dimensions in '{}', with max_dim '{}'".format(dim, max_ndim)
    # assert all(0 <= d < max_ndim for d in ret), "Invalid dimension in '{}', with max_dim '{}'".format(dim, max_ndim)
    return sorted(ret)


def sum(tensor: Tensor, dim: Optional[Union[int, Iterable[int]]] = None, keepdim: bool = False) -> Tensor:
    # CASE: sum up all dim
    if dim is None:
        return torch.sum(tensor)

    # CASE: sum up given dim
    if isinstance(dim, int):
        return torch.sum(tensor, dim=dim, keepdim=keepdim)

    # CASE: sum up multiple dims
    dim = _pos_and_sort_dims(dim, tensor.ndim)
    for d in dim:
        tensor = torch.sum(tensor, dim=d, keepdim=True)
    if not keepdim:
        for i, d in enumerate(dim):
            tensor.squeeze_(d - i)
    return tensor


def mean(tensor: Tensor, dim: Optional[Union[int, Iterable[int]]] = None, keepdim: bool = False) -> Tensor:
    # CASE: mean all dim
    if dim is None:
        return torch.mean(tensor)

    # CASE: mean given dim
    if isinstance(dim, int):
        return torch.mean(tensor, dim=dim, keepdim=keepdim)

    # CASE: mean multiple dims
    dim = _pos_and_sort_dims(dim, tensor.ndim)
    for d in dim:
        tensor = torch.mean(tensor, dim=d, keepdim=True)
    if not keepdim:
        for i, d in enumerate(dim):
            tensor.squeeze_(d - i)
    return tensor


def nelement(tensor: Tensor, dim: Optional[Union[int, Iterable[int]]] = None) -> int:
    """return number of elements of given dims"""
    ret = 1
    # CASE: all elements
    if dim is None:
        return tensor.nelement()

    # CASE: single dim
    if isinstance(dim, int):
        return tensor.shape[dim]

    # CASE: multiple dims
    dim = _pos_and_sort_dims(dim, tensor.ndim)
    for d in dim:
        ret *= tensor.size(d)
    return ret


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                              Spectral Normalization                                              * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def spectral_norm(m):
    if m is None:
        return m

    def fn(m):
        if hasattr(m, "weight"):
            m = nn.utils.spectral_norm(m, name="weight")

    m.apply(fn)
    return m


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      weight_norm                                                     #
# -------------------------------------------------------------------------------------------------------------------- #


def weight_norm(module: nn.Module) -> nn.Module:
    def fn(module):
        if hasattr(module, "weight"):
            module = nn.utils.weight_norm(module)
        return module

    module.apply(fn)
    return module


def remove_weight_norm(module: nn.Module) -> nn.Module:
    def fn(module):
        if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
            module = nn.utils.remove_weight_norm(module)
        return module

    module.apply(fn)
    return module


# -------------------------------------------------------------------------------------------------------------------- #
#                                                    initialization                                                    #
# -------------------------------------------------------------------------------------------------------------------- #


def is_rnn(m: nn.Module) -> bool:
    return isinstance(m, (nn.GRU, nn.LSTM, nn.GRUCell, nn.LSTMCell))


def orthogonal_init(m: nn.Module, gain: float = 1) -> nn.Module:
    assert is_rnn(m)
    for name, param in m.named_parameters():
        if name.find("weight_ih") >= 0:
            nn.init.xavier_uniform_(param, gain=gain)
        elif name.find("weight_hh") >= 0:
            nn.init.orthogonal_(param, gain=gain)
        elif name.find("bias") >= 0:
            nn.init.zeros_(param)
        else:
            raise NameError("unknown param {}".format(name))
    return m


def init_layer(m: nn.Module, init_type: str = "xavier_uniform", **kwargs) -> nn.Module:
    classname = m.__class__.__name__
    if is_rnn(m):
        assert init_type == "orthogonal", f"initialization method [{init_type}] is not implemented for {classname}"
        m = orthogonal_init(m, **kwargs)
    elif hasattr(m, "weight"):
        if not hasattr(nn.init, f"{init_type}_"):
            raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
        # init weight
        fn = getattr(nn.init, f"{init_type}_")
        fn(m.weight, **kwargs)
        # init optional bias
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    else:
        raise NotImplementedError("The type of layer is unknown: {}".format(type(m)))
    return m


# -------------------------------------------------------------------------------------------------------------------- #
#                                                      activation                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def _analyze_activation(activation: str):
    if activation is None:
        activation = "identity"
    if activation.find("leaky_relu?") == 0 or activation.find("lrelu?") == 0:
        # with negative_slope
        args = activation.replace("leaky_relu?", "").replace("lrelu?", "")
        arg_negative_slope = re.match(r"negative_slope=([\d\.]+)", args)
        if arg_negative_slope is None:
            raise ValueError("Invalid '{}'. You may want `leaky_relu?negative_slope=<neg_slope>`".format(activation))
        return "leaky_relu", float(arg_negative_slope.group(1))
    elif re.match(r"(leaky_relu|lrelu)[_-]*([\d\.]+)", activation):
        match = re.match(r"(leaky_relu|lrelu)[_-]*([\d\.]+)", activation)
        return "leaky_relu", float(match.group(2))
    elif activation.find("glu?dim:") == 0:
        return "glu", int(activation[8:])
    else:
        return activation, 0.0


def is_identity_activation(name: str) -> bool:
    return name is None or name.lower() in ["identity", "linear", "none"]


def parse_activation(name: str) -> nn.Module:
    assert name.find("@") < 0, f"Find invalid char '@' in activation name: '{name}'. Should use '?' to pass args"

    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax()
    elif name == "softmax2d":
        return nn.Softmax2d()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softplus":
        return nn.Softplus()
    elif name == "elu":
        return nn.ELU()
    elif is_identity_activation(name):
        return nn.Identity()
    elif name.find("leaky_relu") == 0 or name.find("lrelu") == 0:
        if name == "leaky_relu" or name == "lrelu":
            a = 0.2
        else:
            _, a = _analyze_activation(name)
        return nn.LeakyReLU(negative_slope=a, inplace=True)
    else:
        raise ValueError("Do not support activation of name {}".format(name))


def parse_norm(
    norm: str,
    num_features: int,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    # batch norm
    elif norm in ("bn1", "batch_norm_1d", "BatchNorm1d"):
        return nn.BatchNorm1d(num_features, eps, momentum, affine, True)
    elif norm in ("bn2", "batch_norm_2d", "BatchNorm2d"):
        return nn.BatchNorm2d(num_features, eps, momentum, affine, True)
    elif norm in ("bn3", "batch_norm_3d", "BatchNorm3d"):
        return nn.BatchNorm3d(num_features, eps, momentum, affine, True)
    # instance norm
    elif norm in ("in1", "instance_norm_1d", "InstanceNorm1d"):
        return nn.InstanceNorm1d(num_features, eps, momentum, affine, False)
    elif norm in ("in2", "instance_norm_2d", "InstanceNorm2d"):
        return nn.InstanceNorm2d(num_features, eps, momentum, affine, False)
    elif norm in ("in3", "instance_norm_3d", "InstanceNorm3d"):
        return nn.InstanceNorm3d(num_features, eps, momentum, affine, False)
    # unknown
    else:
        raise NotImplementedError("Unknown norm: {}".format(norm))


# -------------------------------------------------------------------------------------------------------------------- #
#                                                         MISC                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel, nn.parallel.DistributedDataParallelCPU)
    )


class scale_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        grad_out = grad * ctx.scale
        return grad_out, None
