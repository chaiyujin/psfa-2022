import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import wgpu
import wgpu.backends.rs  # noqa: F401, Select Rust backend

_DIR = os.path.dirname(os.path.abspath(__file__))
_DIR_SHADERS = _DIR
SAMPLE_COUNT = 4

flags_type = [
    ("texture_albedo", "int32", [1]),
    ("shading", "int32", [1]),
    ("COOK_BLINN", "int32", [1]),
    ("COOK_BECKMANN", "int32", [1]),
    ("COOK_GGX", "int32", [1]),
]
transform_dtype = [
    ("mat_model", "float32", (4, 4)),
    ("mat_view", "float32", (4, 4)),
    ("mat_proj", "float32", (4, 4)),
    ("mat_norm", "float32", (4, 4)),
]
material_dtype = [
    ("metallic", "float32", [1]),
    ("roughness", "float32", [1]),
    ("rim", "float32", [1]),
]
point_light_dtype = [
    ("misc", "float32", (4,)),
    ("position", "float32", (4,)),
    ("color", "float32", (4,)),
]

clear_color = (0.03, 0.03, 0.03, 0.00)

_mat_opengl_to_wgpu = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class Texture:
    texture: Any
    view: Any
    format: wgpu.Enum


def create_depth_texture(device: wgpu.GPUDevice, w: int, h: int):
    format = wgpu.TextureFormat.depth24plus_stencil8  # type: ignore
    texture = device.create_texture(
        size=(int(w), int(h), 1),
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,  # type: ignore
        dimension=wgpu.TextureDimension.d2,  # type: ignore
        format=format,
        mip_level_count=1,
        sample_count=SAMPLE_COUNT,
    )
    view = texture.create_view()
    return Texture(texture, view, format)


def get_attributes(*which_list: str | tuple[str, int]):
    descs = []
    offset = 0
    for which in which_list:
        desc = {}
        desc["offset"] = offset
        desc["shader_location"] = len(descs)
        if isinstance(which, str):
            if which in ["xyz", "rgb", "normal", "tangent"]:
                dtype, count = "float32", 3
            elif which in ["xyzw", "rgba"]:
                dtype, count = "float32", 4
            elif which in ["uv", "texcoord"]:
                dtype, count = "float32", 2
            else:
                raise ValueError(f"unknown: {which}")
        else:
            dtype, count = which
        desc["format"] = getattr(wgpu.VertexFormat, f"{dtype}x{count}")
        # Append.
        descs.append(desc)

        # Next offset.
        assert dtype == "float32"
        offset += 4 * count
    return descs, offset


vertex_attrs, vertex_stride = get_attributes("xyz", "normal", "uv", "rgb")


class MeshRenderer:
    @classmethod
    def create_offscreen(cls, w: int, h: int):
        from wgpu.gui.offscreen import WgpuCanvas

        return cls(WgpuCanvas(size=(w, h), pixel_ratio=1))

    def __init__(
        self,
        canvas: wgpu.gui.WgpuCanvasBase,
        power_preference: str = "high-performance",
        limits: Any = None,
    ):
        """Regular function to setup a viz on the given canvas."""
        # Note: passing the canvas here can (oddly enough) prevent the
        # adapter from being found. Seen with wx/Linux.
        self.canvas = canvas
        self.adapter = wgpu.request_adapter(canvas=None, power_preference=power_preference)  # type: ignore
        self.device: wgpu.GPUDevice = self.adapter.request_device(required_limits=limits)
        self._inited = False

    @property
    def is_initialized(self):
        return self._inited

    def draw(self) -> npt.NDArray[np.uint8]:
        array = self.canvas.draw()  # type: ignore
        return np.asarray(array)

    def update_transform(
        self,
        mat_model: npt.NDArray[np.float_] | None = None,
        mat_view: npt.NDArray[np.float_] | None = None,
        mat_proj: npt.NDArray[np.float_] | None = None,
        is_opengl: bool = True,
    ):
        def _set_mat(key: str, mat: npt.NDArray[np.float_] | None):
            if mat is not None:
                assert mat.shape == (4, 4)
                # NOTE: WGPU want column first data.
                self._transform_data[key] = mat.astype(np.float32).T
                self._transform_updated = True

        if mat_proj is not None and is_opengl:
            mat_proj = _mat_opengl_to_wgpu @ mat_proj

        _set_mat("mat_model", mat_model)
        _set_mat("mat_view", mat_view)
        _set_mat("mat_proj", mat_proj)
        if mat_model is not None or mat_view is not None:
            m = self._transform_data["mat_model"].T
            v = self._transform_data["mat_view"].T
            _set_mat("mat_norm", np.linalg.inv(m @ v).T)

    def update_vertices(self, data: npt.NDArray[np.float_], part: str | None = None):
        assert len(data) == self.num_verts, f"Invalid {len(data)} data, should be {self.num_verts}"
        if part is None:
            target_size = vertex_stride // 4
            drange = list(range(target_size))
        elif part == "xyz":
            target_size = 3
            drange = [0, 1, 2]
        elif part == "normal":
            target_size = 3
            drange = [3, 4, 5]
        elif part in ["uv", "texcoord", "texcoords"]:
            target_size = 2
            drange = [6, 7]
        elif part in ["rgb", "color"]:
            target_size = 3
            drange = [8, 9, 10]
        else:
            raise NotImplementedError(f"Unknown part: {part}")
        assert (
            data.shape[1] == target_size
        ), f"Invalid shape: {data.shape}, should be ({self.num_verts},{vertex_stride//4})"
        self._vertex_data[:, drange] = data.astype(np.float32)
        self._vertex_updated = True

    def update_indices(self, indices: npt.NDArray[np.uint32]):
        self._index_updated = True
        self._index_data = indices.flatten()

    def toggle(self, which: str, flag: bool | None = None):
        key = which
        if flag is not None:
            new_flag = flag
        else:
            new_flag = not bool(self._flags_data[key][0])
        # Only update when different.
        if self._flags_data[key][0] != int(new_flag):
            # print(self._flags_data[key], new_flag)
            self._flags_data[key][0] = int(new_flag)
            self._flags_updated = True

    def update_texture_albedo(self, tex_data: npt.NDArray[np.uint8]):
        if tex_data.shape[0] != self._texture_size or tex_data.shape[1] != self._texture_size:
            tex_data = cv2.resize(tex_data, (self._texture_size, self._texture_size))
        if tex_data.shape[-1] == 3:
            tex_data = np.pad(tex_data, [(0, 0), (0, 0), (0, 1)], "constant", constant_values=255)
        self._tex_diff_data = np.ascontiguousarray(np.flip(tex_data, axis=0))
        self._tex_diff_updated = True

    def initialize(
        self,
        num_verts: int,
        num_triangles: int,
        shader_source: str | None = None,
        texture_size: int = 256,
    ):
        self.num_verts = num_verts
        self.num_triangles = num_triangles
        device = self.device
        if shader_source is None:
            with open(os.path.join(_DIR_SHADERS, "shader.wgsl")) as fp:
                shader_source = fp.read()
        shader = device.create_shader_module(code=shader_source)

        # Vertex/Index.
        vertex_desc = {
            "array_stride": vertex_stride,
            "step_mode": wgpu.VertexStepMode.vertex,  # type: ignore
            "attributes": vertex_attrs,
        }

        self._vertex_data = np.zeros([self.num_verts, vertex_stride // 4], dtype=np.float32)
        self._vertex_data[..., -3:] = 1.0
        self._index_data = np.arange(self.num_triangles * 3, dtype=np.uint32)
        self._vertex_updated = True
        self._index_updated = True

        vertex_buffer = device.create_buffer(
            size=self._vertex_data.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )
        index_buffer = device.create_buffer(
            size=self._index_data.nbytes,
            usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # [Uniform] flags.
        self._flags_updated = True
        self._flags_data = np.zeros([], dtype=flags_type)
        self._flags_data["shading"] = 1
        self._flags_data["COOK_GGX"][0] = 1
        flags_buffer = device.create_buffer(
            size=self._flags_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # [Uniform] transform.
        self._transform_updated = True
        self._transform_data = np.zeros([], dtype=transform_dtype)
        self._transform_data["mat_model"] = np.eye(4, dtype=np.float32)
        self._transform_data["mat_view"] = np.eye(4, dtype=np.float32)
        self._transform_data["mat_proj"] = np.eye(4, dtype=np.float32)
        self._transform_data["mat_norm"] = np.eye(4, dtype=np.float32)
        transform_buffer = device.create_buffer(
            size=self._transform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # [Uniform] material.
        self._material_update = True
        self._material_data = np.zeros([], dtype=material_dtype)
        self._material_data["metallic"][0] = 0.04
        self._material_data["roughness"][0] = 0.20
        self._material_data["rim"][0] = 0.0
        material_buffer = device.create_buffer(
            size=self._material_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # [Storage] point lights.
        self._point_light_updated = True
        self._point_light_data = np.zeros([5], dtype=point_light_dtype)
        self._point_light_data[0]["misc"][0] = 13.0  # strength
        self._point_light_data[0]["position"] = np.array([0.0, 2.0, 2.0, 1.0], "float32")
        self._point_light_data[0]["color"] = np.array([1.0, 1.0, 1.0, 1.0], "float32")
        self._point_light_data[1]["misc"][0] = 3.0  # strength
        self._point_light_data[1]["position"] = np.array([0.0, 1.0, 2.0, 1.0], "float32")
        self._point_light_data[1]["color"] = np.array([1.0, 1.0, 1.0, 1.0], "float32")

        point_light_buffer = device.create_buffer(
            size=self._point_light_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # [Uniform] sampler.
        sampler = device.create_sampler()

        # [Uniform] texture (albedo_map).
        self._texture_size = texture_size
        d0 = np.ones([texture_size, texture_size, 4], dtype=np.uint8)
        d1 = np.ones([texture_size, texture_size, 4], dtype=np.uint8)
        for i in range(10):
            d0[i::20, :, :] = 0
            d1[:, i::20, :] = 0
        self._tex_diff_data = np.logical_xor(d0, d1).astype(np.uint8) * 255
        self._tex_diff_data[:, :, 3:] = 255
        self._tex_diff_updated = True
        tex_diff = device.create_texture(
            size=(texture_size, texture_size, 1),
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,  # type: ignore
            dimension=wgpu.TextureDimension.d2,  # type: ignore
            format=wgpu.TextureFormat.rgba8unorm,  # type: ignore
            mip_level_count=1,
            sample_count=1,
        )
        tex_diff_view = tex_diff.create_view()

        # Bind groups.
        bind_groups_entries = []
        bind_groups_layout_entries = []

        # group(0): Transform & flags.
        bind_groups_entries.append([])
        bind_groups_layout_entries.append([])
        # > Flags.
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": {
                    "buffer": flags_buffer,
                    "offset": 0,
                    "size": flags_buffer.size,
                },
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "buffer": {"type": wgpu.BufferBindingType.uniform},  # type: ignore
            }
        )
        # > Transform.
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": {
                    "buffer": transform_buffer,
                    "offset": 0,
                    "size": transform_buffer.size,
                },
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "buffer": {"type": wgpu.BufferBindingType.uniform},  # type: ignore
            }
        )
        # > Material.
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": {
                    "buffer": material_buffer,
                    "offset": 0,
                    "size": material_buffer.size,
                },
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "buffer": {"type": wgpu.BufferBindingType.uniform},  # type: ignore
            }
        )
        # > Point lights.
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": {
                    "buffer": point_light_buffer,
                    "offset": 0,
                    "size": point_light_buffer.size,
                },
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},  # type: ignore
            }
        )

        # group(1): Sampler & textures.
        bind_groups_entries.append([])
        bind_groups_layout_entries.append([])
        # > sampler
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": sampler,
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "sampler": {"type": wgpu.SamplerBindingType.filtering},  # type: ignore
            }
        )
        # > tex_diff
        bind_groups_entries[-1].append(
            {
                "binding": len(bind_groups_entries[-1]),
                "resource": tex_diff_view,
            }
        )
        bind_groups_layout_entries[-1].append(
            {
                "binding": len(bind_groups_layout_entries[-1]),
                "visibility": wgpu.ShaderStage.FRAGMENT,  # type: ignore
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,  # type: ignore
                    "view_dimension": wgpu.TextureViewDimension.d2,  # type: ignore
                },
            }
        )

        # Create the wgpu binding objects
        bind_group_layouts = []
        bind_groups = []
        for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
            bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
            bind_group_layouts.append(bind_group_layout)
            bind_groups.append(device.create_bind_group(layout=bind_group_layout, entries=entries))

        # Depth texture
        depth_texture = create_depth_texture(device, *self.canvas.get_logical_size())

        # No bind group and layout, we should not create empty ones.
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)

        present_context = self.canvas.get_context()
        render_texture_format = present_context.get_preferred_format(device.adapter)
        present_context.configure(device=device, format=render_texture_format)
        present_context._sample_count = SAMPLE_COUNT

        render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [vertex_desc],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,  # type: ignore
                "front_face": wgpu.FrontFace.ccw,  # type: ignore
                "cull_mode": wgpu.CullMode.back,  # type: ignore
            },
            depth_stencil={
                "format": depth_texture.format,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,  # type: ignore
            },
            # multisample=None,
            multisample={"count": SAMPLE_COUNT},
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": render_texture_format,
                        "blend": {
                            "color": (
                                wgpu.BlendFactor.one,  # type: ignore
                                wgpu.BlendFactor.zero,  # type: ignore
                                wgpu.BlendOperation.add,  # type: ignore
                            ),
                            "alpha": (
                                wgpu.BlendFactor.one,  # type: ignore
                                wgpu.BlendFactor.zero,  # type: ignore
                                wgpu.BlendOperation.add,  # type: ignore
                            ),
                        },
                    },
                ],
            },
        )

        if SAMPLE_COUNT > 1:
            psize = self.canvas.get_physical_size()
            msaa_texture = device.create_texture(
                label="presentation-context-msaa",
                size=(max(psize[0], 1), max(psize[1], 1), 1),
                format=render_texture_format,
                usage=wgpu.flags.TextureUsage.RENDER_ATTACHMENT,
                sample_count=SAMPLE_COUNT,
            )
            msaa_texture_view = msaa_texture.create_view()
        else:
            msaa_texture_view = None

        def draw_frame():
            current_texture_view = present_context.get_current_texture()
            command_encoder: wgpu.GPUCommandEncoder = device.create_command_encoder()

            # [UNIFORM] Update flag_using_tex.
            if self._flags_updated:
                # Upload the uniform struct
                tmp_buffer_flags = device.create_buffer_with_data(
                    data=self._flags_data,
                    usage=wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(tmp_buffer_flags, 0, flags_buffer, 0, self._flags_data.nbytes)
                self._flags_updated = False

            # [UNIFORM] Update transform.
            if self._transform_updated:
                # Upload the uniform struct
                tmp_buffer_transform = device.create_buffer_with_data(
                    data=self._transform_data,
                    usage=wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(
                    tmp_buffer_transform, 0, transform_buffer, 0, self._transform_data.nbytes
                )
                self._transform_updated = False

            # [UNIFORM] Update material.
            if self._material_update:
                tmp_material_buffer = device.create_buffer_with_data(
                    data=self._material_data,
                    usage=wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(
                    tmp_material_buffer, 0, material_buffer, 0, self._material_data.nbytes
                )
                self._material_update = False

            # [STORAGE] Update point lights.
            if self._point_light_updated:
                tmp_pl_buffer = device.create_buffer_with_data(
                    data=self._point_light_data,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(
                    tmp_pl_buffer, 0, point_light_buffer, 0, self._point_light_data.nbytes
                )
                self._point_light_updated = False

            # Update UNIFORM texture (albedo_map).
            if self._tex_diff_updated:
                device.queue.write_texture(
                    {
                        "texture": tex_diff,
                        "mip_level": 0,
                        "origin": (0, 0, 0),
                    },
                    self._tex_diff_data,
                    {
                        "offset": 0,
                        "bytes_per_row": self._tex_diff_data.strides[0],
                    },
                    (self._tex_diff_data.shape[1], self._tex_diff_data.shape[0], 1),
                )
                self._tex_diff_updated = False

            # Update vertex/index buffer.
            if self._vertex_updated:
                tmp_vb = device.create_buffer_with_data(
                    data=self._vertex_data,
                    usage=wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(tmp_vb, 0, vertex_buffer, 0, self._vertex_data.nbytes)
                self._vertex_updated = False
            if self._index_updated:
                tmp_ib = device.create_buffer_with_data(
                    data=self._index_data,
                    usage=wgpu.BufferUsage.COPY_SRC,  # type: ignore
                )
                command_encoder.copy_buffer_to_buffer(tmp_ib, 0, index_buffer, 0, self._index_data.nbytes)
                self._index_updated = False

            # Render pass.
            if msaa_texture_view is None:
                view = current_texture_view
                resolve_target = None
            else:
                view = msaa_texture_view
                resolve_target = current_texture_view
            render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": view,
                        "resolve_target": resolve_target,
                        "clear_value": clear_color,
                        "load_op": wgpu.LoadOp.clear,  # type: ignore
                        "store_op": wgpu.StoreOp.store,  # type: ignore
                    }
                ],
                depth_stencil_attachment={
                    "view": depth_texture.view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": wgpu.LoadOp.clear,  # type: ignore
                    "depth_store_op": wgpu.StoreOp.store,  # type: ignore
                    "stencil_clear_value": 0,
                    "stencil_load_op": wgpu.LoadOp.clear,  # type: ignore
                    "stencil_store_op": wgpu.StoreOp.store,  # type: ignore
                },
            )
            render_pass.set_pipeline(render_pipeline)
            render_pass.set_vertex_buffer(0, vertex_buffer)
            render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)  # type: ignore
            for bind_group_id, bind_group in enumerate(bind_groups):
                render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
            # render_pass.set
            render_pass.draw_indexed(self._index_data.size, 1, 0, 0, 0)
            render_pass.end()

            # Submit.
            device.queue.submit([command_encoder.finish()])

            # Request next draw.
            self.canvas.request_draw()

        # Set draw function.
        self.canvas.request_draw(draw_frame)
        self._inited = True


if __name__ == "__main__":
    r = MeshRenderer.create_offscreen(480, 480)
    r.initialize(4, 2)

    tex_data = cv2.imread("texture.jpeg")[:, :, [2, 1, 0]]
    tex_data = tex_data[: tex_data.shape[1], :]
    r.update_texture_albedo(tex_data)

    # (x, y, z), (nx, ny, nz), (u, v), (r, g, b)
    r.update_vertices(
        np.array(
            [
                [-0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [+0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [+0.5, +0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                [-0.5, +0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ],
            np.float32,
        )
    )

    # index.
    r.update_indices(
        np.array([[0, 1, 2], [0, 2, 3]], np.uint32).flatten(),
    )

    run()

    cv2.imwrite("triangle_offscreen0.png", r.draw()[..., [2, 1, 0]])

    a1 = time.time()
    r.toggle("texture", True)
    r.update_transform(
        mat_model=np.array(
            [
                [np.cos(a1), -np.sin(a1), 0, 0],
                [np.sin(a1), +np.cos(a1), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
    )
    r._vertex_data[0][0] += 0.1
    r._vertex_updated = True
    cv2.imwrite("triangle_offscreen1.png", r.draw()[..., [2, 1, 0]])

    a1 = time.time() + 2
    r.toggle("texture")
    r.update_transform(
        mat_model=np.array(
            [
                [np.cos(a1), -np.sin(a1), 0, 0],
                [np.sin(a1), +np.cos(a1), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
    )
    r._vertex_data[0][0] += 0.1
    r._vertex_updated = True
    cv2.imwrite("triangle_offscreen2.png", r.draw()[..., [2, 1, 0]])
