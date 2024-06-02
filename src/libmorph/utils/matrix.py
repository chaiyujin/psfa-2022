from typing import Optional

import torch
import torch.nn.functional as F


def _to_tensor(vec):
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec)
    return vec


def _to(vec, dtype, device):
    if (dtype is not None) or (device is not None):
        vec = vec.to(dtype=dtype, device=device)
    return vec, vec.dtype, vec.device


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                    Translation                                                   * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def translate(vec, dtype=None, device=None):
    vec = _to_tensor(vec)
    vec, dtype, device = _to(vec, dtype, device)
    assert vec.shape[-1] == 3

    # chunk components
    _x, _y, _z = vec.chunk(3, dim=-1)
    # stack the matrix
    _0 = torch.zeros_like(_x)
    _1 = torch.ones_like(_x)
    mat = torch.stack(
        [
            torch.cat([_1, _0, _0, _x], dim=-1),
            torch.cat([_0, _1, _0, _y], dim=-1),
            torch.cat([_0, _0, _1, _z], dim=-1),
            torch.cat([_0, _0, _0, _1], dim=-1),
        ],
        dim=-2,
    )
    return mat


def _euler_near_zero_approx(vec):
    _x, _y, _z = vec.chunk(chunks=3, dim=-1)
    nx = _x * -1.0
    ny = _y * -1.0
    nz = _z * -1.0
    _0 = torch.zeros_like(_x)
    _1 = torch.ones_like(_x)

    mat = torch.stack(
        [
            torch.cat([_1, nz, _y, _0], dim=-1),
            torch.cat([_z, _1, nx, _0], dim=-1),
            torch.cat([ny, _x, _1, _0], dim=-1),
            torch.cat([_0, _0, _0, _1], dim=-1),
        ],
        dim=-2,
    )
    return mat


def euler_rotate(vec, eps=1e-8, dtype=None, device=None):
    vec = _to_tensor(vec)
    vec, dtype, device = _to(vec, dtype, device)
    assert vec.shape[-1] == 3

    # get angle
    angle = torch.norm(vec, p=2, dim=-1, keepdim=True)

    # 1. near zero approx
    near_zero_mask = angle <= eps
    mat_approx = _euler_near_zero_approx(vec)

    # 2. to angle-axis
    angle = torch.where(near_zero_mask, torch.full_like(angle, eps), angle)
    axis = vec / angle
    # angle-axis to quat
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axis], dim=-1)
    # quat to mat
    mat_quat = quat2mat(quat, is_normalized=True)

    # 3. merge
    mat = torch.where(near_zero_mask.unsqueeze(-1), mat_approx, mat_quat)

    return mat


def euler2quat(vec, eps=1e-8, dtype=None, device=None):
    vec = _to_tensor(vec)
    vec, dtype, device = _to(vec, dtype, device)
    assert vec.shape[-1] == 3

    # to angle-axis
    angle = torch.norm(vec + eps, p=2, dim=-1, keepdim=True)
    axis = vec / angle

    # angle-axis to quat
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axis], dim=-1)

    return quat


def quat2mat(quat, is_normalized=False, dtype=None, device=None):
    quat = _to_tensor(quat)
    quat, dtype, device = _to(quat, dtype, device)
    assert quat.shape[-1] == 4

    norm_quat = quat
    # not sure quat is normalized or not
    if not is_normalized:
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=-1, keepdim=True)

    w, x, y, z = norm_quat.chunk(4, dim=-1)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    _0 = torch.zeros_like(w2)
    _1 = torch.ones_like(w2)

    # fmt: off
    mat = torch.stack([
        torch.cat([w2 + x2 - y2 - z2,  2 * xy - 2 * wz,    2 * wy + 2 * xz,    _0], dim=-1),
        torch.cat([2 * wz + 2 * xy,    w2 - x2 + y2 - z2,  2 * yz - 2 * wx,    _0], dim=-1),
        torch.cat([2 * xz - 2 * wy,    2 * wx + 2 * yz,    w2 - x2 - y2 + z2,  _0], dim=-1),
        torch.cat([_0,                 _0,                 _0,                 _1], dim=-1),
    ], dim=-2)
    # fmt: on

    return mat


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                 Camera Projection                                                * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def get_persp_projection(intrinsics, zn=0.01, zf=100.0, aspect=None, device=None):
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    assert intrinsics.shape[-1] in [3, 4]
    dtype = intrinsics.dtype
    if device is None:
        device = intrinsics.device
    else:
        intrinsics = intrinsics.to(device)

    if intrinsics.shape[-1] == 3:
        fx, ppx, ppy = intrinsics.chunk(3, dim=-1)
        if aspect is not None:
            fy = fx / aspect
        else:
            fy = fx
    else:
        fx, fy, ppx, ppy = intrinsics.chunk(4, dim=-1)
        if aspect is not None:
            fy = fx / aspect

    a = -(zf + zn) / (zf - zn)
    b = -2.0 * zf * zn / (zf - zn)
    if torch.is_tensor(a):
        a = a.item()
    if torch.is_tensor(b):
        b = b.item()

    # fill constant
    _a = torch.full_like(fx, a, dtype=dtype, device=device)
    _b = torch.full_like(fx, b, dtype=dtype, device=device)
    _c = torch.full_like(fx, -1, dtype=dtype, device=device)
    _0 = torch.full_like(fx, 0, dtype=dtype, device=device)
    _1 = torch.full_like(fx, 1, dtype=dtype, device=device)

    # focus
    _x = fx * 2.0
    _y = fy * 2.0
    # center
    cx = _1 - ppx * 2.0
    cy = ppy * 2.0 - _1

    # matrix
    mat = torch.stack(
        [
            torch.cat([_x, _0, cx, _0], dim=-1),
            torch.cat([_0, _y, cy, _0], dim=-1),
            torch.cat([_0, _0, _a, _b], dim=-1),
            torch.cat([_0, _0, _c, _0], dim=-1),
        ],
        dim=-2,
    )
    return mat


def get_ortho_projection(intrinsics, zn=0.01, zf=100.0, aspect=None, device=None):
    # check inputs and get info
    assert intrinsics.shape[-1] in [3, 4]
    dtype = intrinsics.dtype
    if device is None:
        device = intrinsics.device
    else:
        intrinsics = intrinsics.to(device)

    # get parameters
    if intrinsics.shape[-1] == 3:
        fx, ppx, ppy = intrinsics.chunk(3, dim=-1)
        if aspect is not None:
            fy = fx / aspect
        else:
            fy = fx
    else:
        fx, fy, ppx, ppy = intrinsics.chunk(4, dim=-1)
        if aspect is not None:
            fy = fx / aspect

    # constants
    a = -2.0 / (zf - zn)
    b = -1.0 * (zf + zn) / (zf - zn)
    if torch.is_tensor(a):
        a = a.item()
    if torch.is_tensor(b):
        b = b.item()

    # fill constant
    _a = torch.full_like(fx, a, dtype=dtype, device=device)
    _b = torch.full_like(fx, b, dtype=dtype, device=device)
    _0 = torch.full_like(fx, 0, dtype=dtype, device=device)
    _1 = torch.full_like(fx, 1, dtype=dtype, device=device)

    # focus
    _x = fx * 2.0
    _y = fy * 2.0
    # center
    cx = ppx * 2.0 - 1.0
    cy = 1.0 - ppy * 2.0

    # matrix
    mat = torch.stack(
        [
            torch.cat([_x, _0, _0, cx], dim=-1),
            torch.cat([_0, _y, _0, cy], dim=-1),
            torch.cat([_0, _0, _a, _b], dim=-1),
            torch.cat([_0, _0, _0, _1], dim=-1),
        ],
        dim=-2,
    )
    return mat


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                     Transform vertex positions to clip space                                     * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def transform(mat, pos, normalize: bool = False, eps: Optional[float] = None, pad_value=1):
    assert mat.ndim == 3 and mat.shape[-2] == 4 and mat.shape[-1] == 4, "Invalid matrix shape: {}".format(mat.shape)
    assert pos.ndim == 3 and pos.shape[-1] in [3, 4], "Invalid position shape: {}".format(pos.shape)
    wanna_homo = pos.shape[-1] == 4
    # pad if necessary
    posw = F.pad(pos, (0, 1), "constant", pad_value) if pos.shape[-1] == 3 else pos
    assert (posw[..., -1] == pad_value).all()
    # transform
    posw_clip = torch.bmm(posw, mat.permute(0, 2, 1))
    # normalize
    if normalize:
        # normalize
        denom = posw_clip[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        ret = posw_clip[..., :3] / denom
        if wanna_homo:
            # ret = F.pad(ret, (0, 1), "constant", 1)
            ret = torch.cat((ret, posw_clip[..., 3:]), dim=-1)
    else:
        ret = posw_clip if wanna_homo else posw_clip[..., :3]
    # return
    return ret.contiguous()
