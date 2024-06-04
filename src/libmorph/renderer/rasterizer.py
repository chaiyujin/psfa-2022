from typing import Dict, List, Optional, Tuple, Union

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.attribute import merge_attr_dict, split_attr_dict, vertex_normals
from .cameras import Cameras
from .template import MeshTemplate


class NVDiffRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    GL_CTX = dr.RasterizeGLContext()

    def __init__(self, image_size=256):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if isinstance(image_size, (int, float)):
            self.image_size = (int(image_size), int(image_size))
        else:
            assert isinstance(image_size, (list, tuple)) and len(image_size) == 2
            self.image_size = tuple([int(x) for x in image_size])

    def forward(self, vertices_screen, faces, **kwargs):
        return dr.rasterize(self.GL_CTX, vertices_screen, faces, resolution=self.image_size, **kwargs)


class MeshRasterizer(nn.Module):
    def __init__(self, image_size, template_fpath):
        super().__init__()
        self.image_size = image_size
        self.template = MeshTemplate(template_fpath)
        self.rasterizer = NVDiffRasterizer(image_size)

    def forward(
        self,
        vertices: Tensor,
        vertices_screen: Optional[Tensor] = None,
        cameras: Optional[Cameras] = None,
        eps: Optional[float] = None,
        pixel_attr_names: Union[List[str], Tuple[str, ...]] = ("vertices", "normals", "uvs"),
    ):
        # * transform vertices into screen
        if vertices_screen is None:
            assert cameras is not None, "None of 'vertices_screen' or 'cameras' is given!"
            vertices_homo = F.pad(vertices, (0, 1), "constant", value=1)
            vertices_screen = cameras.transform(vertices_homo, eps=eps)
        else:
            assert cameras is None, "Both 'vertices_screen' and 'cameras' are given!"

        # * fetch template triangles and UV coordinates
        pos_tris: torch.Tensor = self.template.triangles[0]
        uvs_tris: torch.Tensor = self.template.uv_triangles[0]
        uvs: torch.Tensor = self.template.NO_uvcoords

        # * rasterize
        rast_out, _ = self.rasterizer(vertices_screen, pos_tris)
        # masks
        mask_valid = rast_out[..., 3:] > 0

        # fmt: off
        # * get some pixel-space attributes
        pixel_attr_names = set(pixel_attr_names)
        assert all(x in ('vertices', 'normals', 'uvs') for x in pixel_attr_names)

        # * attributes according to pos triangle indices
        pos_attrs = dict()
        if 'vertices' in pixel_attr_names: pos_attrs['vertices'] = vertices
        if 'normals'  in pixel_attr_names: pos_attrs['normals'] = vertex_normals(vertices, pos_tris)
        pix_pos_attrs = self.interpolate_attributes(pos_attrs, rast_out, pos_tris)

        # * attributes according to uv triangle indices
        uvs_attrs = dict()
        if 'uvs' in pixel_attr_names: uvs_attrs['uvs'] = uvs
        pix_uvs_attrs = self.interpolate_attributes(uvs_attrs, rast_out, uvs_tris)
        # fmt: on

        return dict(
            rast_out=rast_out,
            mask_valid=mask_valid,
            vertices_screen=vertices_screen,
            # pixel space: N,H,W,C
            **pix_pos_attrs,
            **pix_uvs_attrs,
        )

    def interpolate_attributes(self, attrs: Dict[str, Tensor], rast_out, tris) -> Dict[str, Tensor]:
        if len(attrs) == 0:
            return dict()
        merged_attrs, kd_list = merge_attr_dict(**attrs)
        pix_attrs, _ = dr.interpolate(merged_attrs, rast_out, tris)
        return split_attr_dict(pix_attrs, kd_list, key_prefix="pixel_")

    def antialias(self, img, rast_out, vertices_screen):
        pos_tris: torch.Tensor = self.template.triangles[0]
        return dr.antialias(img, rast_out, vertices_screen, pos_tris)
