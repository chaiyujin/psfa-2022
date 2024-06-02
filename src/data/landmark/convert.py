from typing import Any, Dict, List, Optional, Tuple

from .preset import PRESET_MAPPING

_indices_dict: Dict[str, Tuple[int, ...]] = dict()


def convert(lmks: Any, src_type: str, dst_type: str) -> Any:
    if src_type == dst_type:
        return lmks

    key = f"{src_type.upper()} -> {dst_type.upper()}"
    # get indices
    if key not in _indices_dict:
        assert key in PRESET_MAPPING, f"Failed to find mapping '{key}' in preset."
        n_points = int(dst_type.split("-")[-1])
        new_indices: List[Optional[int]] = [None for _ in range(n_points)]
        for j, i in PRESET_MAPPING[key].items():
            new_indices[i] = j
        assert not any(
            x is None for x in new_indices
        ), f"Not all points in '{dst_type}' has a source from '{src_type}'!"
        _indices_dict[key] = tuple([(x if x is not None else -1) for x in new_indices])
    indices = _indices_dict[key]
    return lmks[..., indices, :]
