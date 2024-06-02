import pickle
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.engine.logging import get_logger
from src.libmorph import FLAME, build_morphable
from src.modules.layers import MLP

log = get_logger(__name__)


def build_decoder_verts(config, in_ch) -> nn.Module:
    # check if using
    name = config.using
    # alias
    if name in ["transformer"]:
        name = "xfmr"
    # build
    _build_dict = dict(
        morphable=_Morphable,
        blendshape=_BlendShape,
    )
    assert name in _build_dict, f"audio encoder is using unknown module '{name}'"
    m = _build_dict[name](config[name], in_ch)
    return m


class _Morphable(nn.Module):
    def __init__(self, config, in_ch):
        super().__init__()
        self.morphable = build_morphable(config)
        assert isinstance(self.morphable, FLAME)
        self.proj_exp = MLP(in_ch, self.morphable.n_exp, activation="identity")
        self.proj_jaw = MLP(in_ch, 3, activation="identity", init_kwargs={"gain": 0.01})

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if name == "morphable":
                continue
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def forward(self, z, idle_verts):
        exp = self.proj_exp(z)
        jaw = self.proj_jaw(z)
        code_dict = dict(exp=exp, jaw_pose=jaw)
        out_verts = self.morphable(code_dict, v_template=idle_verts)
        return out_verts, code_dict


class _BlendShape(nn.Module):
    def __init__(self, config, in_ch):
        super().__init__()
        self.config = config

        # * basis are trainable or not
        self.trainable = config.trainable

        # * basis list
        using_basis = config.using_basis
        assert using_basis is not None
        if isinstance(using_basis, str):
            using_basis = [using_basis]
        using_basis = [x for x in using_basis if x.lower() not in ["none", "null", "nil"]]
        self.using_basis = using_basis

        # * shared mlp
        self.shared_mlp = MLP(
            in_ch,
            self.config.shared_mlp.out_channels,
            norm_method=self.config.norm_method,
            activation=self.config.activation,
            last_activation=self.config.activation,
        )
        self.latent_channels = self.shared_mlp.in_channels
        decoded_size = self.shared_mlp.out_channels

        # * build all basis modules
        self.coeffs_projectors = nn.ModuleDict()
        self.n_basis = dict()
        for basis_name in using_basis:
            self._load_basis(basis_name, decoded_size)

    def _register_tensor(self, name, ndarray, force_trainable=False):
        if self.trainable or force_trainable:
            self.register_parameter(name, nn.Parameter(torch.tensor(ndarray, dtype=torch.float32)))
        else:
            self.register_buffer(name, torch.tensor(ndarray, dtype=torch.float32))

    def _load_basis(self, basis_name, decoded_size):
        def _build_proj(out_channels):
            return MLP(
                in_channels=decoded_size,
                out_channels=list(self.config.specific_mlp.hidden_channels) + [out_channels],
                norm_method=self.config.norm_method,
                activation=self.config.activation,
                last_activation="identity",
            )

        # fmt: off
        def _load_flame():
            with open(self.config.flame_model_path, "rb") as f:
                flame_model = pickle.load(f, encoding="latin1")

            # The shape components and expression
            shapedirs = to_tensor(to_np(flame_model['shapedirs']), dtype=torch.float32)
            comp = shapedirs[:, :, 300 : 300 + self.config.n_components]
            self._register_tensor(f"comp_from_{basis_name}", comp.view(comp.shape[0]*comp.shape[1], comp.shape[2]))
            self.n_basis[basis_name] = comp.shape[-1]
            self.coeffs_projectors[basis_name] = _build_proj(self.n_basis[basis_name])
            log.info("[BlendShape]: init from FLAME ({} comps)".format(comp.shape))
            assert len(self.using_basis) == 1, "You can only use '{}' by itself".format(basis_name)
        # fmt: on

        if basis_name == "flame":
            _load_flame()
        else:
            raise NotImplementedError("uknown basis: " + basis_name)

    def forward(self, x, idle_verts):
        assert x.ndim == 2
        bsz = x.shape[0]

        latent = self.shared_mlp(x)

        deform_offsets = 0.0
        for basis_name in self.coeffs_projectors:
            if basis_name in ["flame"]:
                coeffs = self.coeffs_projectors[basis_name](latent)
                comp = getattr(self, f"comp_from_{basis_name}")
                offsets = nn.functional.linear(coeffs, comp)
                offsets = offsets.view(bsz, -1, 3)
                # accumulate offsets
                deform_offsets += offsets
            else:
                raise ValueError("Unknown basis: {}".format(basis_name))

        return deform_offsets + idle_verts, {}


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)
