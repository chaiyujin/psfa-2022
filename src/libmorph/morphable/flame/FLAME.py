# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

# * ---------------------------------------------------------------------------
# * This file is adapted from DECA: https://github.com/YadiraF/DECA/
# * 1. Original 'pose' parameter (6-dim) involves (1) rigid pose for entire mesh (2) pose for jaw joint.
# *    Here, they are represented by two seperate paremters, 3-dim each.
# * ---------------------------------------------------------------------------

import logging
import pickle
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils import Struct, to_np, to_tensor
from .lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks

log = logging.getLogger("FLAME")


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config):
        super(FLAME, self).__init__()
        self.cfg = config
        with open(config.model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)
        self.n_shape = self.cfg.n_shape
        self.n_exp = self.cfg.n_exp
        log.info("Creating the FLAME {} shape, {} exp".format(self.n_shape, self.n_exp))

        self.dtype = torch.float32
        self.register_buffer("faces_tensor", to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer("v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, : config.n_shape], shapedirs[:, :, 300 : 300 + config.n_exp]], 2)
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        # Joints
        self.register_buffer("J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype))

    @property
    def triangles(self):
        return self.faces_tensor

    def _check_params(self, params, batch_size, n_params):
        if params is not None:
            assert params.ndim == 2 and params.shape[1] == n_params
            if params.shape[0] != batch_size:
                assert params.shape[0] == 1
                params = params.expand(batch_size, -1)
            return params
        # default value on None
        val = torch.zeros((1, n_params), dtype=self.dtype, device=self.v_template.device)
        return val.expand(batch_size, -1)

    def forward(self, code_dict: Dict[str, Optional[Tensor]], v_template: Optional[Tensor] = None):
        # * Get Parameters
        # fmt: off
        N = 1
        shape     = code_dict.get("shape")
        exp       = code_dict.get("exp")
        neck_pose = code_dict.get("neck_pose")
        jaw_pose  = code_dict.get("jaw_pose")
        eye_pose  = code_dict.get("eye_pose")

        if shape     is not None: N = max(N, shape.shape[0])
        if exp       is not None: N = max(N, exp.shape[0])
        if neck_pose is not None: N = max(N, neck_pose.shape[0])
        if jaw_pose  is not None: N = max(N, jaw_pose.shape[0])
        if eye_pose  is not None: N = max(N, eye_pose.shape[0])

        shape     = self._check_params(shape,     N, self.cfg.n_shape)
        exp       = self._check_params(exp,       N, self.cfg.n_exp)
        neck_pose = self._check_params(neck_pose, N, 3)
        jaw_pose  = self._check_params(jaw_pose,  N, 3)
        eye_pose  = self._check_params(eye_pose,  N, 6)
        # fmt: on

        if v_template is None:
            v_template = self.v_template.unsqueeze(0)
        assert v_template.ndim == 3
        if v_template.shape[0] == 1 and v_template.shape[0] != N:
            v_template = v_template.expand(N, -1, -1)

        # * LBS
        betas = torch.cat([shape, exp], dim=1)
        morph_pose = torch.cat([neck_pose, jaw_pose, eye_pose], dim=1)
        vertices, _ = lbs(
            betas,
            torch.cat([torch.zeros_like(jaw_pose), morph_pose], dim=1),
            v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )
        return vertices
