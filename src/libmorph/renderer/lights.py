import numpy as np
import torch
import torch.nn.functional as F


def add_lights(light_type, lights, **kwargs):
    if light_type == "point":
        return add_point_lights(lights.unsqueeze(1), **kwargs)
    elif light_type == "sh9":
        return add_sh9_lights(lights, **kwargs)
    else:
        raise NotImplementedError()


def add_point_lights(lights, pixel_vertices, pixel_normals, shininess=16.0, **kwargs):
    """
    Args:
        vertices: [bz, h, w, 3]
        normals:  [bz, h, w, 3]
        lights:   [bz, nlight, 9]
    Returns:
        shading:  [bz, h, w, 3]
    """
    assert lights.ndim == 3 and lights.shape[-1] == 9
    assert pixel_vertices.ndim == 4 and pixel_vertices.shape[-1] == 3
    assert pixel_normals.ndim == 4 and pixel_normals.shape[-1] == 3
    shape = pixel_normals.shape[:-1]
    vertices = pixel_vertices.view(shape[0], shape[1] * shape[2], 3)  # bsz, n_pixels, 3
    normals = pixel_normals.view(shape[0], shape[1] * shape[2], 3)  # bsz, n_pixels, 3

    light_pos = lights[:, :, :3]
    light_diff = lights[:, :, 3:6]
    light_spec = lights[:, :, 6:9]

    # diffuse
    # - light_dirs: bsz, n_lights, n_pixels, 3
    light_dirs = F.normalize(light_pos[:, :, None, :] - vertices[:, None, :, :], dim=3, eps=1e-6)
    #  - dot: bsz, n_lights, n_pixels
    diff = torch.relu((normals[:, None, :, :] * light_dirs).sum(dim=3))
    diff_shading = (diff[:, :, :, None] * light_diff[:, :, None, :]).sum(1)  # bsz, n_pixels, 3

    # specular
    # - assume viewer is (0,0,0)
    view_pos = torch.zeros_like(light_pos)
    view_dirs = F.normalize(view_pos[:, :, None, :] - vertices[:, None, :, :], dim=3, eps=1e-6)
    halfway = F.normalize(light_dirs + view_dirs, dim=3, eps=1e-6)
    # - spec
    spec = torch.relu((normals[:, None, :, :] * halfway).sum(dim=3))
    spec = torch.pow(spec, shininess)
    spec_shading = (spec[:, :, :, None] * light_spec[:, :, None, :]).sum(1)  # bsz, n_pixels, 3

    shading = diff_shading + spec_shading
    return shading.view(*shape, 3)


def add_direction_lights(lights, pixel_normals, **kwargs):
    """
    Args:
        normals: [bz, h, w, 3]
        lights:  [bz, nlight, 6]
    Returns:
        shading: [bz, h, w, 3]
    """
    assert lights.ndim == 3 and lights.shape[-1] == 6
    assert pixel_normals.ndim == 4 and pixel_normals.shape[-1] == 3
    shape = pixel_normals.shape[:-1]
    normals = pixel_normals.view(shape[0], shape[1] * shape[2], 3)

    light_direction = lights[:, :, :3]
    light_intensities = lights[:, :, 3:]
    directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
    normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
    shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
    shading = shading.sum(1)

    return shading.view(*shape, 3)


def add_sh9_lights(sh_coeffs, pixel_normals, **kwargs):
    """
    Args:
        sh_coeff: [bz, 9, 3]
    """
    N = pixel_normals.permute(0, 3, 1, 2)
    sh = torch.stack(
        [
            N[:, 0] * 0.0 + 1.0,
            N[:, 0],
            N[:, 1],
            N[:, 2],
            N[:, 0] * N[:, 1],
            N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2],
            N[:, 0] ** 2 - N[:, 1] ** 2,
            3 * (N[:, 2] ** 2) - 1,
        ],
        1,
    )  # [bz, 9, h, w]
    factors = torch.tensor(SH_CONST_FACTORS, dtype=sh.dtype, device=sh.device)
    sh = sh * factors[None, :, None, None]
    # import ipdb; ipdb.set_trace()
    shading = torch.sum(sh_coeffs[:, :, :, None, None] * sh[:, :, None, :, :], 1)
    return shading.permute(0, 2, 3, 1)


SH_CONST_FACTORS = np.array(
    [
        1 / np.sqrt(4 * np.pi),
        ((2 * np.pi) / 3) * (np.sqrt(3 / (4 * np.pi))),
        ((2 * np.pi) / 3) * (np.sqrt(3 / (4 * np.pi))),
        ((2 * np.pi) / 3) * (np.sqrt(3 / (4 * np.pi))),
        (np.pi / 4) * (3) * (np.sqrt(5 / (12 * np.pi))),
        (np.pi / 4) * (3) * (np.sqrt(5 / (12 * np.pi))),
        (np.pi / 4) * (3) * (np.sqrt(5 / (12 * np.pi))),
        (np.pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * np.pi))),
        (np.pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * np.pi))),
    ],
    dtype=np.float32,
)
