import torch
import torch.nn.functional as F

from src.data.mesh.torch.utils import face_normals
from src.engine.graphics import GL_CTX, dr, matrix


def naive_render_mesh(pos, tri, col=None, shading="flat", image_size=(512, 512), **kwargs):
    # check inputs
    assert pos.ndim == 3 and pos.shape[-1] == 3
    assert tri.ndim == 2 and tri.shape[-1] == 3
    assert shading in ["flat", "smooth", "none", None]
    bsz = pos.shape[0]
    device = pos.device

    # get matrices
    mat_proj = kwargs.get("mat_proj")
    if mat_proj is None:
        mat_proj = matrix.get_persp_projection([[1.5, 1.5, 0.5, 0.5]], device=device).expand(bsz, -1, -1)
    else:
        mat_proj = mat_proj.to(device)

    mat_view = kwargs.get("mat_view")
    if mat_view is None:
        mat_view = torch.eye(4, device=device).unsqueeze(0).expand(bsz, -1, -1)
    else:
        mat_view = mat_view.to(device)

    mat_model = kwargs.get("mat_model")
    if mat_model is None:
        mat_model = matrix.translate([[0, 0, -0.4]], device=device).expand(bsz, -1, -1)
    else:
        mat_model = mat_model.to(device)

    assert mat_proj.shape[0] == bsz, f"mat_proj ({mat_proj.shape}) has wrong shape, should {bsz} batch_size"
    assert mat_view.shape[0] == bsz, f"mat_view ({mat_view.shape}) has wrong shape, should {bsz} batch_size"
    assert mat_model.shape[0] == bsz, f"mat_model ({mat_model.shape}) has wrong shape, should {bsz} batch_size"
    # print("proj", mat_proj[0])
    # print("view", mat_view[0])
    # print("model", mat_model[0])

    # get indices
    pos_idx = tri
    tri_idx = torch.arange(len(pos_idx) * 3, device=device, dtype=torch.int).view(-1, 3)

    # transform positions
    mat_pvm = torch.bmm(mat_proj, torch.bmm(mat_view, mat_model))
    pos_homo = torch.nn.functional.pad(pos, (0, 1), "constant", value=1)
    pos_clip = matrix.transform(mat_pvm, pos_homo)
    # flip y axis
    pos_clip = pos_clip * torch.tensor([1, -1, 1, 1], device=device)

    # rasterize
    rast_out, _ = dr.rasterize(GL_CTX, pos_clip, pos_idx, resolution=image_size)

    if col is not None:
        pix_col, _ = dr.interpolate(col, rast_out, pos_idx)
    else:
        pix_col = 1.0

    if shading is not None and shading != "none":
        # prepare normals
        tri_normals = face_normals(pos, tri, flat=(shading == "flat")).view(bsz, -1, 3)
        # object coord -> world coord
        # mat_normal = torch.inverse(mat_model)
        mat_normal = mat_model
        nrm_world = matrix.transform(mat_normal, tri_normals, normalize=False, pad_value=0)
        nrm_world = F.normalize(nrm_world, dim=-1)
        pos_world = matrix.transform(mat_model, pos, normalize=False)
        # interpolate attributes into pixels
        pix_pos, _ = dr.interpolate(pos_world, rast_out, pos_idx)
        pix_normals, _ = dr.interpolate(nrm_world, rast_out, tri_idx)
        pix_normals = F.normalize(pix_normals, dim=-1, eps=1e-6)
        # - Guess the size of mesh
        vmax = pos.abs().max()
        # - Preset point lights
        point_lights = torch.tensor(
            [
                [
                    [vmax * -1.8, vmax * +2.2, vmax * 5.5, 0.80, 0.80, 0.80, 0.10, 0.10, 0.10],
                    [vmax * +5.5, vmax * -1.4, vmax * 5.5, 0.21, 0.28, 0.34, 0.05, 0.05, 0.05],
                ]
            ],
            device=device,
        ).repeat(bsz, 1, 1)
        point_lights[..., :3] = matrix.transform(mat_model, point_lights[..., :3], normalize=False)
        # - Shading with point lights
        shading = add_point_lights(pix_pos, pix_normals, point_lights)

        # Shade with color
        color = shading * pix_col

        # color = color ** (1.0/1.8)

        # # Back face culling
        # z_positive = torch.tensor([[[[0, 0, 1]]]], device=device, dtype=torch.float)
        # back_face_test = (pix_normals * z_positive).sum(3, keepdim=True)
        # color = torch.where(back_face_test >= 0, color, torch.zeros_like(color))

    else:
        assert col is not None
        color = pix_col

    # TODO: texture sampling

    return color


def add_point_lights(pixel_vertices, pixel_normals, lights, shininess=16.0):
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
