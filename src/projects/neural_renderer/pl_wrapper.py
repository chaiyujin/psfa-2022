from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim

from src import metrics
from src.engine import ops
from src.engine.train import get_optim_and_lrsch, print_dict, reduce_dict, sum_up_losses
from src.modules.neural_renderer import NeuralRenderer
from src.projects.anim import PL_AnimBase, SeqInfo

from . import visualizer
from .gan_loss import GANLoss, NLayerDiscriminator
from .vgg_loss import VGGLoss


class PL_Wrapper(PL_AnimBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # rendering module
        self.renderer = NeuralRenderer(self.hparams.neural_renderer)
        self.seq_info = SeqInfo(1)

        # vgg loss module
        self.vgg_loss = VGGLoss()

        # (optional) GAN loss module
        self.gan_loss = None
        self.net_D = None
        if self.hparams.gan_mode not in [None, "none"]:
            self.gan_loss = GANLoss(self.hparams.gan_mode)
            self.net_D = NLayerDiscriminator(input_nc=3, norm_method="instance")
            # NOTE: close the automatic_optimization for GAN
            self.automatic_optimization = False

    def final_nr_fake(self, results):
        return results["nr_fake"]

    def forward(self, batch, **kwargs):
        trck_delta = batch["offsets_tracked"]
        idle_verts = batch["idle_verts"].unsqueeze(1)
        mesh_verts = trck_delta + idle_verts

        prob: float = float(np.random.uniform()) if self.hparams.aug_crop else 10.0
        aug_args: Optional[List[Tuple[int, int, int]]] = None
        A = batch["image"].shape[-1]
        crop_size = A // 16
        if self.training and prob < 0.5:
            aug_args = []
            new_target = []
            new_mouth_masks = []
            for bi in range(len(batch["image"])):
                x = np.random.randint(0, crop_size + 1)  # type: ignore
                y = np.random.randint(0, crop_size + 1)  # type: ignore
                a = np.random.randint(A - crop_size * 2, A - max(x, y) + 1)  # type: ignore
                aug_args.append((x, y, a))
                # target
                tar = batch["image"][bi : bi + 1, 0, :, y : y + a, x : x + a]
                tar = F.upsample(tar, size=(A, A), mode="bilinear", align_corners=True)
                new_target.append(tar.unsqueeze(1))
                # mouth mask
                mask = batch["mouth_mask"][bi : bi + 1, 0, :, y : y + a, x : x + a].float()
                mask = F.upsample(mask, size=(A, A), mode="bilinear", align_corners=True) > 0.05
                new_mouth_masks.append(mask.unsqueeze(1))
            batch["image"] = torch.cat(new_target)
            batch["mouth_mask"] = torch.cat(new_mouth_masks)

        flip_dims = None
        if self.training and self.hparams.aug_flip:
            # aug_flip = [np.random.uniform() < 0.5, np.random.uniform() < 0.5]  x, y
            aug_flip = [np.random.uniform() < 0.5]  # x
            flip_dims = [-(i + 1) for i, x in enumerate(aug_flip) if x]
            if len(flip_dims) > 0:
                batch["image"] = torch.flip(batch["image"], dims=flip_dims)
                batch["mouth_mask"] = torch.flip(batch["mouth_mask"].float(), dims=flip_dims).bool()

        return self.render(
            mesh_verts, idle_verts, batch["code_dict"], batch["clip_source"], batch["image"], aug_args, flip_dims
        )

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                Actual Forward                                                * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def render(self, verts, idle_verts, code_dict, clip_sources, target, aug_args, flip_dims):
        rotat = code_dict["rotat"]
        transl = code_dict["transl"]
        cam = code_dict["cam"]

        assert verts.shape[1] == 1, "invalid shape: {}".format(verts.shape)
        assert rotat.shape[1] == 1, "invalid shape: {}".format(rotat.shape)
        assert transl.shape[1] == 1, "invalid shape: {}".format(transl.shape)
        assert cam.shape[1] == 1, "invalid shape: {}".format(cam.shape)
        assert target.shape[1] == 1, "invalid shape: {}".format(target.shape)

        ret = self.renderer(
            clip_sources,
            verts=verts.squeeze(1),
            idle_verts=idle_verts.squeeze(1),
            rotat=rotat.squeeze(1),
            transl=transl.squeeze(1),
            cam=cam.squeeze(1),
            background=target.squeeze(1),
            aug_args=aug_args,
            flip_dims=flip_dims,
        )
        for key in ret:
            ret[key] = ret[key].unsqueeze(1)
        return ret

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                   Optimizer                                                  * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def configure_optimizers(self):
        if self.gan_loss is None:
            optim, lrsch = get_optim_and_lrsch(self.hparams.optim, self.renderer.parameters())
            return [optim], [lrsch]
        else:
            assert self.net_D is not None
            optim_g, lrsch_g = get_optim_and_lrsch(self.hparams.optim, self.renderer.parameters())
            optim_d, lrsch_d = get_optim_and_lrsch(self.hparams.optim_d, self.net_D.parameters())
            return [optim_g, optim_d], [lrsch_g, lrsch_d]

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                            Training and validation                                           * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def training_step(self, batch, batch_idx):
        if self.gan_loss is None:
            return self.training_step_nogan(batch)
        else:
            return self.training_step_gan(batch)

    def training_step_nogan(self, batch):
        assert self.automatic_optimization
        # make sure train mode
        self.train()

        # get render dict
        render_dict = self(batch)
        loss, loss_log = self.get_loss(batch, render_dict, training=True)

        # log vars
        if self.hparams.debug and self.global_step % 10 == 0:
            print_dict("loss", loss_log)
        self.log_dict(loss_log, prog_bar=True)

        # log images
        gap_epoch = 10 if self.hparams.debug else self.hparams.visualizer.draw_gap_steps
        if gap_epoch > 0 and self.global_step % gap_epoch == 0:
            N = min(3, render_dict["nr_fake"].shape[0])
            plt_kwargs = dict(n_plot=N, batch=batch, results=render_dict, debug_mode="rows")
            for painter in ["draw_nr_inter", "draw_nr"]:
                self.tb_add_images("train", painter=painter, **plt_kwargs)

        return dict(loss=loss, items_log=loss_log)

    def training_step_gan(self, batch):
        assert not self.automatic_optimization
        assert self.net_D is not None
        assert self.gan_loss is not None
        self.train()
        self.zero_grad()
        optim_g, optim_d = self.optimizers()  # type: ignore
        logs = dict()

        # rendering forward
        render_dict = self(batch)

        # get fake, real
        fake = render_dict["nr_fake"].squeeze(1)
        real = batch["image"].squeeze(1)
        # print(fake.shape, real.shape)

        # optimize D
        def _backward_D():
            # zero grad
            optim_d.zero_grad()  # type: ignore
            # fake must detach, prevent gradients flows back
            item_fake = self.gan_loss(self.net_D(fake.detach()), False)  # type: ignore
            item_real = self.gan_loss(self.net_D(real), True)  # type: ignore
            # combine loss and calculate gradients
            loss_d = (item_fake + item_real) * 0.5
            logs.update({"loss/gan:D": loss_d.item()})
            # setp
            self.manual_backward(loss_d)
            optim_d.step()

        def _backward_G():
            # zero grad
            optim_g.zero_grad()  # type: ignore
            # supervision
            loss_sup, logs_sup = self.get_loss(batch, render_dict, training=True)
            logs.update(logs_sup)
            # adverserial
            loss_adv = self.gan_loss(self.net_D(fake), True)  # type: ignore
            logs.update({"loss/gan:G_adv": loss_adv.item()})
            # combine
            loss_g = loss_sup + loss_adv * self.hparams.loss.scale_adv
            # step
            self.manual_backward(loss_g)
            optim_g.step()

        _backward_D()
        _backward_G()

        # log vars
        if self.hparams.debug and self.global_step % 10 == 0:
            print_dict("loss", logs)
        self.log_dict(logs, prog_bar=True)

        # log images
        gap_epoch = 10 if self.hparams.debug else self.hparams.visualizer.draw_gap_steps
        if gap_epoch > 0 and self.global_step % gap_epoch == 0:
            N = min(3, render_dict["nr_fake"].shape[0])
            plt_kwargs = dict(n_plot=N, batch=batch, results=render_dict, debug_mode="rows")
            for painter in ["draw_nr_inter", "draw_nr"]:
                self.tb_add_images("train", painter=painter, **plt_kwargs)

        return dict(items_log=logs)

    def validation_step(self, batch, batch_idx):
        # make sure eval mode
        self.eval()

        render_dict = self(batch)
        loss, loss_log = self.get_loss(batch, render_dict, training=False)
        metric_log = self.get_metrics(batch, render_dict, prefix="val_metric")

        logs = {**metric_log, **loss_log}
        self.log_dict(logs)

        return dict(val_loss=loss, items_log=logs)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                     Loss                                                     * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_loss(self, batch, results, training):

        loss_opts = self.hparams.loss
        loss_dict, wght_dict = dict(), dict()

        # get some tensor and check shape
        real, mask, mask_innm = batch["image"], results["mask"], batch["mouth_mask"]
        assert real.shape[-3] == 3
        assert list(mask.shape[-2:]) == list(real.shape[-2:]) and mask.shape[-3] == 1
        assert list(mask_innm.shape[-2:]) == list(real.shape[-2:]) and mask_innm.shape[-3] == 1

        def _mask_weight(mask, mask_innm):
            assert mask.ndim == 5
            assert mask_innm.ndim == 5
            mask_size = float(np.prod(mask.shape[2:]))
            mask_scale = mask_size / torch.clamp(ops.sum(mask.float(), dim=[2, 3, 4], keepdim=True), min=1)
            mask_weight = mask.float() + mask_innm.float() * (loss_opts.scale_inner_mouth - 1)
            return mask_weight * mask_scale

        def _masked(img, mask):
            return torch.where(mask, img, real)

        mask_weight = _mask_weight(mask, mask_innm)

        # item: full image
        if loss_opts.scale_fake > 0:
            fake = results["nr_fake"]
            item = F.l1_loss(fake, real, reduction="none")
            weights = torch.ones_like(mask_innm).float() + mask_innm.float() * (loss_opts.scale_inner_mouth - 1)
            assert item.ndim == 5 and weights.ndim == 5
            loss_dict["nr:fake"] = (item * weights).mean()
            wght_dict["nr:fake"] = loss_opts.scale_fake

            if loss_opts.scale_fake_vgg > 0:
                item_vgg = self.vgg_loss(fake, real)
                loss_dict["nr:fake_vgg"] = item_vgg
                wght_dict["nr:fake_vgg"] = loss_opts.scale_fake * loss_opts.scale_fake_vgg

        # item: face only (masked)
        if loss_opts.scale_face > 0:
            fake = _masked(results["nr_fake_inter"], mask)
            item = F.l1_loss(fake, real, reduction="none")
            assert item.ndim == 5
            loss_dict["nr:face"] = (item * mask_weight).mean()
            wght_dict["nr:face"] = loss_opts.scale_face

            if loss_opts.scale_face_vgg > 0:
                item_vgg = self.vgg_loss(fake, real)
                loss_dict["nr:face_vgg"] = item_vgg
                wght_dict["nr:face_vgg"] = loss_opts.scale_face * loss_opts.scale_face_vgg

        # item: face tex3 only (masked)
        if loss_opts.scale_tex3 > 0:
            fake = _masked(results["nr_tex3"], mask)
            item = F.l1_loss(fake, real, reduction="none")
            assert item.ndim == 5
            loss_dict["nr:tex3"] = (item * mask_weight).mean()
            wght_dict["nr:tex3"] = loss_opts.scale_tex3

            if loss_opts.scale_tex3_vgg > 0:
                item_vgg = self.vgg_loss(fake, real)
                loss_dict["nr:tex3_vgg"] = item_vgg
                wght_dict["nr:tex3_vgg"] = loss_opts.scale_tex3 * loss_opts.scale_tex3_vgg

        # sum up
        loss, logs = sum_up_losses(loss_dict, wght_dict, training=training)

        return loss, logs

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                                    Metrics                                                   * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def get_metrics(self, batch, results, reduction="mean", prefix=None):

        # get logs
        log = dict()
        log["img:psnr"] = metrics.psnr(results["nr_fake"], batch["image"])
        log["img:ssim"] = metrics.ssim(results["nr_fake"], batch["image"])
        log["img:psnr(masked)"] = metrics.psnr_masked(results["nr_fake"], batch["image"], results["mask"])
        log["img:ssim(masked)"] = metrics.ssim_masked(results["nr_fake"], batch["image"], results["mask"])
        log = reduce_dict(log, reduction=reduction, detach=True)

        prefix = "" if prefix is None else (prefix + "/")
        return {prefix + k: v.detach() for k, v in log.items()}


# if loss_opts.scale_zoomed_mouth > 0 and training:
#     # masked_innm = _masked(results["nr_fake"], mask_innm)
#     masked_fake = _masked(results["nr_fake"], mask)
#     _, zoom_fake, zoom_real = _zoomed_mouth_area(mask_innm, masked_fake, target)
#     item = F.l1_loss(zoom_fake, zoom_real, reduction="none")
#     assert item.ndim == 5
#     scale = loss_opts.scale_zoomed_mouth
#     # loss_dict["nr:fake_zoom_mouth"] = (item * _mask_weight_only(zoom_mask)).mean()
#     loss_dict["nr:fake_zoom_mouth"] = item.mean()
#     wght_dict["nr:fake_zoom_mouth"] = scale
#     loss_dict["nr:fake_zoom_mouth_vgg"] = self.vgg_loss(zoom_fake, zoom_real)
#     wght_dict["nr:fake_zoom_mouth_vgg"] = scale * loss_opts.scale_zoomed_mouth_vgg
#
#
# def _zoomed_mouth_area(mask, *img_list):
#     N, L, H, W, _ = mask.shape
#     h, w = int(H // 3), int(W // 3)
#
#     mask = mask.view(N * L, H, W, 1)
#     xy_list = []
#     for m in mask:
#         nonzero = torch.nonzero(m)
#         if nonzero.shape[0] == 0:
#             x = (W - w) // 2
#             y = (H - h) // 2 + H // 6
#         else:
#             y0, y1 = int(nonzero[:, 0].min()), int(nonzero[:, 0].max())
#             x0, x1 = int(nonzero[:, 1].min()), int(nonzero[:, 1].max())
#             yc, xc = (y0 + y1) // 2, (x0 + x1) // 2
#             y = max(min(yc + h // 2, H) - h, 0)
#             x = max(min(xc + w // 2, W) - w, 0)
#         xy_list.append((x, y))
#
#     def _zoom_in(img):
#         C = img.shape[-1]
#         if img.ndim == 5:
#             img = img.view(N * L, H, W, C)
#         assert img.ndim == 4
#         new_list = []
#         for i in range(N * L):
#             x, y = xy_list[i]
#             new_im = img[i : i + 1, y : y + h, x : x + w, :].permute(0, 3, 1, 2)
#             new_im = F.upsample(new_im, mode="bilinear", size=(H, W), align_corners=True)
#             new_im = new_im.permute(0, 2, 3, 1).view(1, H, W, C)
#             new_list.append(new_im)
#         new_img = torch.cat(new_list)
#         return new_img.view(N, L, H, W, C)
#
#     new_img_list = [_zoom_in(mask.float())]
#     for img in img_list:
#         new_img = _zoom_in(img)
#         new_img_list.append(new_img)
#
#     # import cv2
#     # rows = []
#     # for r in range(3):
#     #     cv_im_list = [x[r, 0].detach().cpu().numpy() * 0.5 + 0.5 for x in new_img_list]
#     #     cv_im_list[0] = cv_im_list[0].repeat(3, -1)
#     #     rows.append(np.concatenate(cv_im_list, axis=1)[..., [2, 1, 0]])
#     # cv2.imshow('debug_zoom', np.concatenate(rows))
#     # cv2.waitKey(1)
#     return new_img_list
#
#
# def _debug_im(fake, real):
#     import cv2
#
#     cvim0 = fake[0, 0].detach().cpu().numpy() / 2 + 0.5
#     cvim1 = real[0, 0].cpu().numpy() / 2 + 0.5
#     cvcnvs = np.concatenate((cvim0, cvim1), axis=1)
#     cv2.imshow("im", cvcnvs[..., [2, 1, 0]])  # type: ignore
#     cv2.waitKey(1)
