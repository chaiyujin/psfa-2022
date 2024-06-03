import os

_backend = os.environ.get("MIKU_RENDER_BACKEND", "WGPU")

if _backend == "FILAMENT":
    from .backend_filament import *
elif _backend == "VULKAN":
    from .backend_vulkan import *
elif _backend == "WGPU":
    from .backend_wgpu import *
else:
    raise NotImplementedError(f"Invalid value for 'MIKU_RENDER_BACKEND': {_backend}")
