import torch

_lip_pidx = list(range(46, 64))


def lmd_lip(fake, real, reduction):
    assert fake.shape[-2] in [73, 75]
    assert real.shape[-2] in [73, 75]

    fake = fake[..., _lip_pidx, :2]
    real = real[..., _lip_pidx, :2]

    dist = torch.norm(fake - real, p=2, dim=-1)
    if reduction == "mean":
        dist = dist.mean(-1)
    elif reduction == "max":
        dist, _ = dist.max(-1)

    # import numpy as np
    # import cv2
    # canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # for p in fake[0, 0]:
    #     c = (
    #         int((p[0] + 1) / 2 * canvas.shape[1]),
    #         int((p[1] + 1) / 2 * canvas.shape[0]),
    #     )
    #     cv2.circle(canvas, c, 3, (0, 0, 255))
    # for p in real[0, 0]:
    #     c = (
    #         int((p[0] + 1) / 2 * canvas.shape[1]),
    #         int((p[1] + 1) / 2 * canvas.shape[0]),
    #     )
    #     cv2.circle(canvas, c, 3, (0, 255, 0))
    # cv2.imshow('lmks', canvas)
    # cv2.waitKey(1)

    return dist


def lmd_inner(fake, real, reduction):
    assert fake.shape[-2] in [73, 75]
    assert real.shape[-2] in [73, 75]

    fake = fake[..., 15:65, :2]
    real = real[..., 15:65, :2]
    dist = torch.norm(fake - real, p=2, dim=-1)
    if reduction == "mean":
        dist = dist.mean(-1)
    elif reduction == "max":
        dist, _ = dist.max(-1)

    # import numpy as np
    # import cv2
    # canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    # for p in fake[0, 0]:
    #     c = (
    #         int((p[0] + 1) / 2 * canvas.shape[1]),
    #         int((p[1] + 1) / 2 * canvas.shape[0]),
    #     )
    #     cv2.circle(canvas, c, 3, (0, 0, 255))
    # for p in real[0, 0]:
    #     c = (
    #         int((p[0] + 1) / 2 * canvas.shape[1]),
    #         int((p[1] + 1) / 2 * canvas.shape[0]),
    #     )
    #     cv2.circle(canvas, c, 3, (0, 255, 0))
    # cv2.imshow('lmks', canvas)
    # cv2.waitKey(1)

    return dist
