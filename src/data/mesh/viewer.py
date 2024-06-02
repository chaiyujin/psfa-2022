import numpy as np
import torch

from .numpy import Mesh

_tmpl_v = None
_tmpl_f = None


def set_template(v=None, f=None, filename=None):
    global _tmpl_v
    global _tmpl_f

    if filename is not None:
        assert v is None and f is None
        m = Mesh.LoadFile(filename)
        v, f = m.v, m.f
    else:
        assert v is not None and f is not None

    _tmpl_v = v
    _tmpl_f = f


def set_texture(*args, **kwargs):
    pass


def render(
    v, f=None, vertex_colors=None, shading="flat", image_size=(512, 512), dtype="float32", output_format="rgb", **kwargs
) -> np.ndarray:
    return np.zeros(image_size + (3,), dtype=np.uint8)

    from .torch.renderer import naive_render_mesh

    if f is None:
        f = _tmpl_f
    assert f is not None, "'f' is not given and template is not set!"
    assert dtype in ["float32", "uint8", np.float32, np.uint8]
    assert output_format in ["rgb", "bgr"]
    assert v.ndim in [2, 3] and v.shape[-1] == 3
    assert f.ndim == 2 and f.shape[-1] == 3
    ndim = v.ndim

    # to torch
    c = vertex_colors
    if not torch.is_tensor(v):
        v = torch.from_numpy(v)
    if not torch.is_tensor(f):
        f = torch.from_numpy(f)
    if c is not None and not torch.is_tensor(c):
        c = torch.from_numpy(c)
    for k in ("mat_model", "mat_view", "mat_proj"):
        if kwargs.get(k) is not None and not torch.is_tensor(kwargs[k]):
            kwargs[k] = torch.from_numpy(kwargs[k])

    # expand dim
    if ndim == 2:
        v = v.unsqueeze(0)
        c = c.unsqueeze(0) if c is not None else None
    for k in ("mat_model", "mat_view", "mat_proj"):
        if kwargs.get(k) is not None and kwargs[k].ndim == 2:
            kwargs[k] = kwargs[k].unsqueeze(0)

    # cuda
    v = v.float().cuda()
    f = f.int().cuda()  # type: ignore
    c = c.float().cuda() if c is not None else None
    for k in ("mat_model", "mat_view", "mat_proj"):
        if kwargs.get(k) is not None:
            kwargs[k] = kwargs[k].float().cuda()

    # naive render
    with torch.no_grad():
        img = naive_render_mesh(v, f, col=c, shading=shading, image_size=image_size, **kwargs)
        img = img.detach().cpu().numpy()  # type: ignore
        img = np.clip(img, 0, 1)

    if ndim == 2:
        img = img[0]
    if output_format == "bgr":
        img = np.ascontiguousarray(img[..., [2, 1, 0]])
    if dtype == "uint8" or dtype == np.uint8:
        img = (img * 255).astype(dtype)

    return img
