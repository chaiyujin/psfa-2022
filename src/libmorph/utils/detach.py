from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import ListConfig


class CODES(object):
    # fmt: off
    ALL    = ("shape", "exp", "neck_pose", "jaw_pose", "eye_pose", "rotat", "transl", "cam", "tex", "sh9")
    NON_ID = (         "exp", "neck_pose", "jaw_pose", "eye_pose", "rotat", "transl", "cam",        "sh9")
    ID     = ("shape",                                                                       "tex",      )
    # fmt: on


def _get_detach_list(detach) -> Tuple[str, ...]:
    if isinstance(detach, (tuple, list, ListConfig)):
        return tuple(detach)
    elif isinstance(detach, str):
        if detach.startswith("CODES_"):
            assert hasattr(CODES, detach[6:]), f"Invalid detach value: '{detach}'"
            return getattr(CODES, detach[6:])
        else:
            return (detach,)
    else:
        return detach


def detach_codes(code_dict, detach, camera_type):
    if detach is None:
        detach = []
    detach = _get_detach_list(detach)

    # ! detach transl for ortho camera
    if camera_type.lower().startswith("ortho"):
        detach = detach + ("transl",)
    assert all(x in CODES.ALL for x in detach)

    # * return new dict, to avoid modify original dict
    ret = dict()
    for key, val in code_dict.items():
        if val is None:
            continue
        if key in detach:
            ret[key] = val.detach()
        elif key == "cam" and camera_type.lower().startswith("persp"):
            # ! detach cam shift for perspective camera
            ret[key] = torch.cat((val[..., :-2], val[..., -2:].detach()), dim=-1)
        else:
            ret[key] = val
    return ret, detach
