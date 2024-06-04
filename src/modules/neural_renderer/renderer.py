import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from assets import ASSETS_ROOT, get_selection_obj, get_selection_vidx
from src.libmorph import MeshRasterizer, OrthographicCameras, PerspectiveCameras, Rigid, build_extension
from src.libmorph.config import ExtensionConfig
from src.modules.pix2pix import Pix2Pix
from src.modules.unet import UNetRenderer

from . import utils
from .textures import DynamicTextures


def build_camera(camera_type, camera_aspect=1.0):
    if camera_type.lower().startswith("persp"):
        return PerspectiveCameras(camera_aspect=camera_aspect)
    elif camera_type.lower().startswith("ortho"):
        return OrthographicCameras(camera_aspect=camera_aspect)
    else:
        raise NotImplementedError(f"[build_camera]: '{camera_type}' is unknown")


def build_renderer(
    renderer_type: str,
    input_nc: int,
    output_nc: int,
    ngf: int = 64,
    norm_method: str = "instance",
    dropout: float = 0.1,
) -> nn.Module:
    if renderer_type.lower().startswith("unet"):
        return UNetRenderer(
            renderer=renderer_type,
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            use_dropout=dropout > 0,
            dropout=dropout,
            use_norm=(norm_method not in [None, "none"]),
            norm_method=norm_method,
            last_activation=nn.Tanh,
        )
    elif renderer_type.lower() == "pix2pix":
        return Pix2Pix(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            n_local_enhancers=1,
            n_local_residual_blocks=3,
            n_global_downsampling=3,
            n_global_residual_blocks=3,
            norm_method=norm_method,
            padding_type="reflect",
            activation="relu",
        )
    else:
        raise NotImplementedError("Unknown renderer: {}".format(renderer_type))


def _sampling(textures, uvs):
    bsz = uvs.size(0)
    assert uvs.ndim == 4
    assert uvs.shape[3] == 2
    if textures.shape[0] != uvs.shape[0]:
        assert textures.shape[0] == 1
        textures = textures.expand(bsz, -1, -1, -1)

    sampled_texture = F.grid_sample(textures, uvs, mode="bilinear", align_corners=False)
    tex3 = sampled_texture[:, :3]
    return sampled_texture, tex3


def _fill_mouth(masks, dilating=0):
    npy_masks = masks.cpu().numpy().astype(np.float32)
    kernel = np.ones((dilating, dilating), np.uint8) if dilating > 0 else None
    filled_masks = []
    for mask in npy_masks:
        assert mask.ndim == 3 and mask.shape[-1] == 1
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        filled_mask = utils.fill_hole(mask)
        if kernel is not None:
            filled_mask = cv2.dilate(filled_mask, kernel, iterations=1)
        filled_masks.append(filled_mask.astype(np.float32) / 255.0)
    return torch.tensor(filled_masks, device=masks.device) == 1.0


def _aug_tensor(tensor, aug_args):
    if aug_args is None:
        return tensor

    assert len(tensor) == len(aug_args)
    assert 1 <= tensor.shape[1] <= 3
    A = tensor.shape[-1]
    new_list = []
    for bi in range(len(aug_args)):
        x, y, a = aug_args[bi]
        tar = tensor[bi : bi + 1, :, y : y + a, x : x + a]
        tar = F.upsample(tar, size=(A, A), mode="nearest")
        new_list.append(tar)
    new_tensor = torch.cat(new_list)
    return new_tensor


class NeuralRenderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert len(config.clip_sources) == 1
        assert len(config.speakers) == 1

        self.inpaint_concat = config.inpaint_concat
        self.inpaint_dilated_only = config.inpaint_dilated_only
        self._dilating_jaw_factor = config.dilating_jaw_factor
        self._dilating_kernel_size = config.dilating_kernel_size

        self._build_face_renderer(config)
        self._build_inpainter(config)

    def _build_face_renderer(self, cfg):
        # Build modules
        self.rigid = Rigid()
        self.camera = build_camera(cfg.camera_type, cfg.camera_aspect)
        self.rasterizer = MeshRasterizer(cfg.image_size, get_selection_obj("face2"))
        self.vidx = get_selection_vidx("face2")

        # Extension of flame for uv masking
        self.extension = build_extension(ExtensionConfig())

        # Inner mouth
        self.speaker = cfg.speakers[0]
        innm_dir = os.path.join(ASSETS_ROOT, "innm", self.speaker)
        self.rast_mouth = MeshRasterizer(cfg.image_size, os.path.join(innm_dir, "innm.obj"))
        A, B, lmk_vidx, lmk_wght = utils.things_for_morphing_inner_mouth(innm_dir)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("lmk_vidx", lmk_vidx)
        self.register_buffer("lmk_wght", lmk_wght)

        # Build neural textures
        self.tex_face = DynamicTextures(cfg, cfg.tex_features, n_filters=1, use_rotat=cfg.tex_face_cond_rotat)
        self.tex_mths = DynamicTextures(cfg, cfg.tex_features, n_filters=1)  # , warping=True)

        # Build neural renderer
        self.face_renderer = build_renderer(
            renderer_type=cfg.renderer_type,
            input_nc=cfg.tex_features,
            output_nc=3,
            ngf=cfg.ngf,
            norm_method=cfg.norm if cfg.use_norm else "none",
            dropout=cfg.dropout if cfg.use_dropout else 0.0,
        )

    def _build_inpainter(self, cfg):
        self._inpainter = build_renderer(
            renderer_type=cfg.inpainter_type,
            input_nc=3 + 3 * int(self.inpaint_concat),
            output_nc=3,
            ngf=cfg.ngf,
            norm_method=cfg.norm if cfg.use_norm else "none",
            dropout=cfg.dropout if cfg.use_dropout else 0.0,
        )
        # jaw open delta to enlarge mask
        self.register_buffer("jaw_open_offsets", utils.get_flame_jaw_open().unsqueeze(0))

    def forward(
        self,
        _,
        verts: Tensor,
        idle_verts: Tensor,
        rotat: Tensor,
        transl: Tensor,
        cam: Tensor,
        background: Tensor,
        aug_args: Optional[List[Tuple[int, int, int]]] = None,
        flip_dims: Optional[List[int]] = None,
    ):
        # render face
        inter_dict = self.render_face(_, verts, idle_verts, rotat, transl, cam, aug_args, flip_dims)
        mask_face = inter_dict["mask"]
        fake_face = inter_dict["nr_fake_inter"]

        # mask out face
        background, mask_dilated = self._mask_out_face(verts, rotat, transl, cam, background, aug_args, flip_dims)

        # merge
        if self.inpaint_concat:
            fake_face = torch.where(mask_face, fake_face, torch.zeros_like(fake_face))
            fake_merged = torch.cat((fake_face, background), dim=1)
        else:
            fake_merged = torch.where(mask_face, fake_face, background)

        # inpaint
        inpainted = self._inpainter(fake_merged)  # NCHW

        # only get the inpainted area
        if self.inpaint_dilated_only:
            mask_inpaint = torch.logical_and(mask_dilated, torch.logical_not(mask_face))
            fake_face_dilated = torch.where(mask_inpaint, inpainted, fake_face)
            fake = torch.where(mask_dilated, fake_face_dilated, background)
            # # debug
            # def _write_cv_img(name, tensor):
            #     im = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
            #     im = (im * 0.5 + 0.5) * 255.0
            #     im = np.clip(im, 0, 255). astype(np.uint8)
            #     cv2.imwrite(name, im[..., [2, 1, 0]])
            # _write_cv_img("debug-mask_face.png", mask_face)
            # _write_cv_img("debug-mask_dilated.png", mask_dilated)
            # _write_cv_img("debug-mask_inpainted.png", mask_inpaint)
            # _write_cv_img("debug-fake_face.png", fake_face)
            # _write_cv_img("debug-background.png", background)
            # _write_cv_img("debug-inpainted.png", inpainted)
            # _write_cv_img("debug-fake_face_dilated.png", fake_face_dilated)
            # _write_cv_img("debug-fake.png", fake)
            # quit()
        else:
            fake = inpainted

        return dict(
            nr_fake=fake,
            nr_fake_inter=inter_dict["nr_fake_inter"],
            mask=inter_dict["mask"],
            mask_innm=inter_dict["mask_innm"],
            nr_tex3=inter_dict["nr_tex3"],
            nr_tex_face=inter_dict["nr_tex_face"],
            nr_tex_mouth=inter_dict["nr_tex_mouth"],
        )

    def render_face(self, _, verts, idle_verts, rotat, transl, cam, aug_args=None, flip_dims=None):
        v_face = verts[..., self.vidx, :]
        v_mths = self._morph_inner_mouth(verts, idle_verts)
        rast_face = self._rasterize(v_face, rotat, transl, cam, self.rasterizer)
        rast_mths = self._rasterize(v_mths, rotat, transl, cam, self.rast_mouth)

        # * uv masking
        # - mask of face
        mask_face = self._get_uv_mask(rast_face, "face_weye")  # no mouth
        mask_face_filled = _fill_mouth(mask_face)  # including inner mouth
        # - mask of inner mouth
        mask_mths = torch.logical_and(rast_mths["mask_valid"], mask_face_filled)
        mask_mths = torch.logical_and(mask_mths, torch.logical_not(mask_face))
        # - final mask
        mask = torch.logical_or(mask_face, mask_mths)

        # * uv
        uvs_face = torch.where(mask_face, rast_face["pixel_uvs"], torch.full_like(rast_face["pixel_uvs"], -2))
        uvs_mths = torch.where(mask_mths, rast_mths["pixel_uvs"], torch.full_like(rast_mths["pixel_uvs"], -2))

        # mask reshape
        mask = mask.squeeze(-1).unsqueeze(1)  # N1HW
        mask_mths = mask_mths.squeeze(-1).unsqueeze(1)  # N1HW

        # * augment
        if aug_args is not None:
            mask = _aug_tensor(mask.float(), aug_args) > 0.999
            mask_mths = _aug_tensor(mask_mths.float(), aug_args) > 0.999
            uvs_face = _aug_tensor(uvs_face.permute(0, 3, 1, 2), aug_args).permute(0, 2, 3, 1)
            uvs_mths = _aug_tensor(uvs_mths.permute(0, 3, 1, 2), aug_args).permute(0, 2, 3, 1)
        if flip_dims is not None and len(flip_dims) > 0:
            mask = torch.flip(mask.float(), flip_dims).bool()
            mask_mths = torch.flip(mask_mths.float(), flip_dims).bool()
            uvs_face = torch.flip(uvs_face, [x - 1 for x in flip_dims])  # NHW2
            uvs_mths = torch.flip(uvs_mths, [x - 1 for x in flip_dims])

        # * texture sampling
        tex_face = self.tex_face(v_face, rot=rotat)
        tex_mths = self.tex_mths(v_face, rot=rotat)
        smp_tex_face, nr_tex3_face = _sampling(tex_face, uvs_face)
        smp_tex_mths, nr_tex3_mths = _sampling(tex_mths, uvs_mths)
        # merge
        smp_tex = torch.where(mask_mths, smp_tex_mths, smp_tex_face)  # NCHW
        nr_tex3 = torch.where(mask_mths, nr_tex3_mths, nr_tex3_face)  # NCHW

        # * render
        nr_fake = self.face_renderer(smp_tex)  # NCHW
        # mask
        nr_fake = torch.where(mask, nr_fake, torch.zeros_like(nr_fake))
        nr_tex3 = torch.where(mask, nr_tex3, torch.zeros_like(nr_tex3))

        return dict(
            nr_fake_inter=nr_fake,
            nr_tex3=nr_tex3,
            mask=mask,
            mask_innm=mask_mths,
            nr_tex_face=tex_face,
            nr_tex_mouth=tex_mths,
        )

    def _rasterize(self, v, rot, tsl, cam, rast):
        v = self.rigid(v, rotat=rot, transl=tsl)
        v = F.pad(v, (0, 1), "constant", value=1)
        v_scrn = self.camera.transform(v, intrinsics=cam)
        return rast(v[..., :3], v_scrn, pixel_attr_names=("uvs",))

    def _morph_inner_mouth(self, verts, idle_verts):
        # get delta
        delta_innm = utils.get_inner_mouth_offsets(
            verts, idle_verts, self.A, self.B, self.lmk_vidx, self.lmk_wght, self.speaker
        )
        innm_verts = self.rast_mouth.template.vertices + delta_innm
        return innm_verts

    def _get_uv_mask(self, rast_dict, name):
        rast_mask = rast_dict["mask_valid"]
        uvs = rast_dict["pixel_uvs"]
        uv_mask = self.extension.uv_mask(name, uvs.shape[0])
        pix_uv_masks = F.grid_sample(uv_mask, uvs, mode="nearest")[:, 0, ..., None]
        return torch.logical_and(pix_uv_masks == 1.0, rast_mask)

    def _mask_out_face(self, verts, rotat, transl, cam, background, aug_args=None, flip_dims=None):
        verts = verts + self.jaw_open_offsets * self._dilating_jaw_factor  # type: ignore
        verts = verts[..., self.vidx, :]
        rast_out = self._rasterize(verts, rotat, transl, cam, self.rasterizer)
        mask = self._get_uv_mask(rast_out, "face_weye")  # no mouth
        mask = _fill_mouth(mask, dilating=self._dilating_kernel_size)  # dilate the face mask
        mask = mask.squeeze(-1).unsqueeze(1)

        if aug_args is not None:
            mask = _aug_tensor(mask.float(), aug_args) > 0.999
        if flip_dims is not None and len(flip_dims) > 0:
            mask = torch.flip(mask.float(), flip_dims).bool()

        background = torch.where(mask, torch.zeros_like(background), background)
        return background, mask
