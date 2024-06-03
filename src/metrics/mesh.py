import torch


def _indices_from_axis(axis):
    idx = []
    # fmt: off
    for k in axis:
        if 'x' == k: idx.append(0)
        elif 'y' == k: idx.append(1)
        elif 'z' == k: idx.append(2)
        else:
            raise ValueError('unknown {}'.format(k))
    # fmt: on
    return idx


def verts_dist(fake, real, vidx=None, axis=None, reduction="mean"):
    assert fake is not None and real is not None

    # # debug
    # if vidx is not None:
    #     import cv2
    #     from src.data.mesh.io import load_mesh
    #     from src.data.mesh.viewer import render
    #     from assets import get_vocaset_template_triangles
    #     tris = get_vocaset_template_triangles()

    #     n_vidx = [x for x in range(5023) if x not in vidx]
    #     v_fake = fake[0, 0].detach().cpu().numpy()
    #     v_real = real[0, 0].detach().cpu().numpy()
    #     v_fake[..., n_vidx, :] = 0
    #     v_real[..., n_vidx, :] = 0
    #     im_fake = render(v_fake, tris)
    #     im_real = render(v_real, tris)
    #     cv2.imshow("im_fake", im_fake)
    #     cv2.imshow("im_real", im_real)
    #     cv2.waitKey()
    #     # quit()

    assert fake.shape == real.shape
    assert fake.ndim >= 2 and fake.shape[-1] == 3 and fake.shape[-2] == 5023
    # select vertices
    if vidx is not None:
        fake = fake[..., vidx, :]
        real = real[..., vidx, :]
    # select axis
    if axis is not None:
        cidx = _indices_from_axis(axis)
        fake = fake[..., cidx]
        real = real[..., cidx]
    # dist
    diff = (real - fake).detach()
    dist = torch.norm(diff, p=2, dim=-1)
    if reduction in ["mean", "avg"]:
        dist = dist.mean(-1)
    elif reduction == "max":
        dist, _ = dist.max(-1)
    else:
        raise ValueError("unknown reduction: {}".format(reduction))
    return dist
