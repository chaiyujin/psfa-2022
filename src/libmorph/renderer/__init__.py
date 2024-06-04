from .cameras import OrthographicCameras, PerspectiveCameras
from .flame_tex import FLAMETex
from .rasterizer import MeshRasterizer
from .sh9_shader import SH9Shader


def build_texture(config):
    if config.using_texture == "flame_tex":
        return FLAMETex(config.flame_tex)
    else:
        raise NotImplementedError(f"[build_texture]: '{config.using_texture}' is unknown.")


def build_camera(config):
    if config.camera_type.lower().startswith("persp"):
        return PerspectiveCameras(camera_aspect=config.camera_aspect)
    elif config.camera_type.lower().startswith("ortho"):
        return OrthographicCameras(camera_aspect=config.camera_aspect)
    else:
        raise NotImplementedError(f"[build_camera]: '{config.camera_type}' is unknown")


def build_rendering_related(config):
    camera = build_camera(config)
    rasterizer = MeshRasterizer(config.image_size, config.template_fpath)
    tex_module = build_texture(config)

    if config.using_texture == "flame_tex":
        renderer = SH9Shader()
    else:
        raise NotImplementedError(f"[build_renderer]: unknown texture type '{config.using_texture}'")

    return camera, rasterizer, tex_module, renderer
