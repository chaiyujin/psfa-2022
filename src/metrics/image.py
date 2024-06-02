from typing import Dict, Optional

import numpy as np
import torch

from src.engine.logging import get_logger

from .pytorch_ssim import SSIM

# from torchmetrics import SSIM
# _ssim_module = SSIM(reduction="none", data_range=1.0)
_ssim_module = SSIM(size_average=False)
logger = get_logger("METRICS")


class PSNR_255:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    @staticmethod
    def compute(img1, img2):
        # the last 3 channels should be image!
        assert img1.ndim >= 3

        mse = ((img1 - img2) ** 2).mean(-1).mean(-1).mean(-1)
        return 10 * torch.log10(65025.0 / mse)


def _masked_average(value, mask):
    assert value.shape[-1] == mask.shape[-1]
    assert value.shape[-2] == mask.shape[-2]
    assert value.shape[-3] == 3 and mask.shape[-3] == 1

    masked_value = torch.where(mask, value, torch.zeros_like(value))
    masked_sum = masked_value.mean(-3).sum(-1).sum(-1)
    count = mask.float().mean(-3).sum(-1).sum(-1)
    return masked_sum / torch.clamp(count, 1e-8)


class MaskedPSNR_255:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    @staticmethod
    def compute(img1, img2, mask):
        # the last 3 channels should be image!
        assert img1.ndim >= 3
        assert img1.ndim == mask.ndim

        se = (img1 - img2) ** 2
        mse = _masked_average(se, mask)

        return 10 * torch.log10(65025.0 / mse)


def _masked(img, mask):
    return torch.where(mask, img, torch.zeros_like(img))


def _get_masked_imgs(imgs, mask):
    if imgs.ndim == 4:
        imgs = imgs[:, None, ...]
    if mask.ndim == 4:
        mask = mask[:, None, ...]
    assert imgs.ndim == 5
    assert mask.ndim == 5
    if imgs.shape[-1] in [1, 3, 4]:
        imgs = imgs.permute(0, 1, 4, 2, 3)  # N,L,C,H,W
    if mask.shape[-1] in [1, 3, 4]:
        mask = mask.permute(0, 1, 4, 2, 3)  # N,L,C,H,W
    return _masked(imgs, mask)


def _to_CHW(imgs):
    if imgs.ndim == 4:
        imgs = imgs[:, None, ...]
    assert imgs.ndim == 5
    if imgs.shape[-1] in [1, 3, 4]:
        imgs = imgs.permute(0, 1, 4, 2, 3)  # N,L,C,H,W
    return imgs


def _convert_range(fake, real, vmax):
    assert real.shape == fake.shape, "input image shape: {}, batch image shape: {}".format(fake.shape, real.shape)
    fake = (fake + 1.0) * vmax / 2.0  # 0~vmax
    real = (real + 1.0) * vmax / 2.0  # 0~vmax
    return fake, real


def psnr(fake, real) -> Optional[torch.Tensor]:
    if fake is None or real is None:
        return None

    fake = _to_CHW(fake)
    real = _to_CHW(real)
    fake, real = _convert_range(fake, real, 255.0)
    m_tensor = PSNR_255.compute(fake, real)

    # import cv2
    # for bi in range(len(fake)):
    #     for fi in range(len(fake[bi])):
    #         my_psnr = m_tensor[bi, fi].item()
    #         cv_fake = fake[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_real = real[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_psnr = cv2.PSNR(cv_fake, cv_real, 255.0)
    #         print(cv_psnr, my_psnr)
    #         assert np.isclose(cv_psnr, my_psnr), "PSNR: CV {} != MY {}".format(cv_psnr, my_psnr)
    #         cv2.imshow('psnr - fake', cv_fake / 255.0)
    #         cv2.imshow('psnr - real', cv_real / 255.0)
    #         cv2.waitKey(1)

    return m_tensor


def psnr_masked(fake, real, mask) -> Optional[torch.Tensor]:
    if fake is None or real is None or mask is None:
        return None

    if mask.shape[-1] in [1, 3, 4]:
        mask = mask.permute(0, 1, 4, 2, 3)  # N,L,C,H,W

    fake = _get_masked_imgs(fake, mask)
    real = _get_masked_imgs(real, mask)
    fake, real = _convert_range(fake, real, 255.0)
    m_tensor = MaskedPSNR_255.compute(fake, real, mask)

    # m_tensor = PSNR_255.compute(fake, real)
    # print(m_tensor[0, 0])
    # import cv2
    # for bi in range(len(fake)):
    #     for fi in range(len(fake[bi])):
    #         my_psnr = m_tensor[bi, fi].item()
    #         cv_fake = fake[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_real = real[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_psnr = cv2.PSNR(cv_fake, cv_real, 255.0)
    #         print(cv_psnr, my_psnr)
    #         # assert np.isclose(cv_psnr, my_psnr), "PSNR: CV {} != MY {}".format(cv_psnr, my_psnr)
    #         cv2.imshow('psnr_masked - fake', cv_fake / 255.0)
    #         cv2.imshow('psnr_masked - real', cv_real / 255.0)
    #         cv2.waitKey(1)

    return m_tensor


# ==================================================================================================================== #
# SSIM                                                                                                                 #
# ==================================================================================================================== #


def ssim(fake, real) -> Optional[torch.Tensor]:
    if fake is None or real is None:
        return None

    fake = _to_CHW(fake)
    real = _to_CHW(real)

    bsz, frm = real.shape[:2]
    fake, real = _convert_range(fake, real, 1.0)
    m_tensor = _ssim_module(fake.view(bsz * frm, *fake.shape[2:]), real.view(bsz * frm, *real.shape[2:]))
    m_tensor = m_tensor.view(bsz, frm, *m_tensor.shape[1:])

    m_tensor = m_tensor.mean(-3).mean(-1).mean(-1)

    # import cv2
    # from skimage.metrics import structural_similarity as ssim
    # for bi in range(len(fake)):
    #     for fi in range(len(fake[bi])):
    #         my_ssim = m_tensor[bi, fi].item()
    #         cv_fake = fake[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_real = real[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_ssim = ssim(cv_fake, cv_real, win_size=11, multichannel=True, gaussian_weights=True)
    #         print(cv_ssim, my_ssim)
    #         cv2.imshow('ssim - fake', cv_fake)
    #         cv2.imshow('ssim - real', cv_real)
    #         cv2.waitKey(1)

    return m_tensor


def ssim_masked(fake, real, mask) -> Optional[torch.Tensor]:
    if fake is None or real is None or mask is None:
        return None

    if mask.shape[-1] in [1, 3, 4]:
        mask = mask.permute(0, 1, 4, 2, 3)  # N,L,C,H,W

    fake = _get_masked_imgs(fake, mask)
    real = _get_masked_imgs(real, mask)

    bsz, frm = real.shape[:2]
    fake, real = _convert_range(fake, real, 1.0)
    m_tensor = _ssim_module(fake.view(bsz * frm, *fake.shape[2:]), real.view(bsz * frm, *real.shape[2:]))
    m_tensor = m_tensor.view(bsz, frm, *m_tensor.shape[1:])

    m_tensor = _masked_average(m_tensor, mask)

    # import cv2
    # from skimage.metrics import structural_similarity as ssim
    # for bi in range(len(fake)):
    #     for fi in range(len(fake[bi])):
    #         my_ssim = m_tensor[bi, fi].item()
    #         cv_fake = fake[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_real = real[bi, fi].permute(1, 2, 0).detach().cpu().numpy()
    #         cv_ssim = ssim(cv_fake, cv_real, win_size=11, multichannel=True, gaussian_weights=True)
    #         print(cv_ssim, my_ssim)
    #         cv2.imshow('ssim_masked - fake', cv_fake)
    #         cv2.imshow('ssim_masked - real', cv_real)
    #         cv2.waitKey(1)

    return m_tensor
