import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import toml
import torch


def normalize(
    lmks: Union[torch.Tensor, np.ndarray], normalize_with: Union[Tuple[Any, ...], torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:

    if isinstance(normalize_with, (tuple, list)):
        assert len(normalize_with) == 2 and all(
            x not in [1, 2, 3, 4] for x in normalize_with
        ), f"[landmark.normalize]: possibly normalize with wrong shape: {normalize_with}"
        if torch.is_tensor(lmks):
            normalize_with = torch.tensor(normalize_with, dtype=lmks.dtype, device=lmks.device)
        else:
            normalize_with = np.asarray(normalize_with, dtype=lmks.dtype)
    else:
        assert (
            (normalize_with.ndim == lmks.ndim or normalize_with.ndim == 1)
            and normalize_with.shape[-1] == 2
            and (normalize_with > 4).all()
        ), (
            f"[landmark.normalize]: possibly normalize with wrong shape: "
            f"{normalize_with}, the landmarks's shape is {lmks.shape}"
        )
    return (lmks * 2.0 / normalize_with) - 1.0


def denormalize(
    normalized_lmks: Union[
        torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor], Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]
    ],
    denormalize_with: Union[Tuple[Any, ...], torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:

    # convert lmks
    if not isinstance(normalized_lmks, (tuple, list)):
        lmks: Union[torch.Tensor, np.ndarray] = normalized_lmks
    else:
        if torch.is_tensor(normalized_lmks[0]):
            lmks = torch.stack(normalized_lmks)
        else:
            lmks = np.asarray(normalized_lmks)

    # convert denormalize_with
    if isinstance(denormalize_with, (tuple, list)):
        assert len(denormalize_with) == 2 and all(
            x not in [1, 2, 3, 4] for x in denormalize_with
        ), f"[landmark.normalize]: possibly denormalize with wrong shape: {denormalize_with}"
        if torch.is_tensor(lmks):
            denormalize_with = torch.tensor(denormalize_with, dtype=lmks.dtype, device=lmks.device)
        else:
            denormalize_with = np.asarray(denormalize_with, dtype=lmks.dtype)
    else:
        assert (
            (denormalize_with.ndim == lmks.ndim or denormalize_with.ndim == 1)
            and denormalize_with.shape[-1] == 2
            and (denormalize_with > 4).all()
        ), (
            f"[landmark.normalize]: possibly denormalize with wrong shape: "
            f"{denormalize_with}, the landmarks's shape is {lmks.shape}"
        )

    # compute
    return (lmks + 1.0) * denormalize_with / 2.0


def load(
    fpath: str,
    required_type: Optional[str] = None,
    required_n_points: Optional[int] = None,
    normalize_with: Optional[Union[Tuple[Any, ...], np.ndarray]] = None,
) -> np.ndarray:
    """Load landmarks from file
    Args:
        fpath (str) : the file path, can be '*.pts' or '*.toml'
        required_type (str), `optional`: the type of landmarks
        required_n_points: (int), `optional`: the number of points in each frame.
    Return:
        landamrks (np.ndarray or None): in shape (N_Points, Dim) or (N_Frame, N_Points, Dim).
    """

    _read_fns = {".pts": read_pts, ".toml": read_toml}

    ext = os.path.splitext(fpath)[1]
    assert ext in _read_fns, "[landamrk.load]: unknown extension of file {}".format(fpath)

    lmks, info = _read_fns[ext](fpath)
    if normalize_with is not None:
        lmks = normalize(lmks, normalize_with)

    # check required information
    if required_type is not None and info.get("type") is not None:
        assert (
            info["type"].lower() == required_type.lower()
        ), f"Loaded landmarks have wrong type '{info['type']}', '{required_type}' is required ({fpath})"
    if required_n_points is not None:
        assert (
            required_n_points == lmks.shape[-2]
        ), f"Loaded landmarks have wrong n_points '{lmks.shape[-2]}', '{required_n_points}' is required ({fpath})"

    # return
    return lmks


def read_pts(fpath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    frames: List[np.ndarray] = []
    points: List[List[float]] = []
    info_dict: Dict[str, Any] = dict(type="unknown")

    def _line_info(i_line, line):
        info = line.split(":")
        assert len(info) == 2, f"Failed to parse line {i_line}: {line} ({fpath})"
        if info[0] == "n_points":
            if info_dict.get("n_points") is not None:
                raise ValueError(f"Define multiple 'n_points' ({fpath})")
            info_dict[info[0]] = int(info[1])
        else:
            info_dict[info[0]] = info[1]

    def _line_brace(i_line, line):
        if line[0] == "{":
            assert info_dict.get("n_points") is not None, f"No 'n_points' is given before points ({fpath})"
            points.clear()
        elif line[0] == "}":
            assert (
                len(points) == info_dict["n_points"]
            ), f"Only {len(points)} are given, should be {info_dict['n_points']} ({fpath})"
            # append into frames
            frames.append(np.asarray(points, dtype=np.float32))
            points.clear()

    def _line_point(i_line, line):
        pt = line.split()
        try:
            points.append([float(x) for x in pt])
        except Exception:
            raise Exception(f"Failed to parse line {i_line}: {line} ({fpath})")

    with open(fpath) as fp:
        for i_line, line in enumerate(fp):
            line = line.strip()

            # empty line or comments
            if len(line) == 0 or line[:2] == "//":
                continue
            # valid lines
            if line.find(":") >= 0:
                _line_info(i_line, line)
            elif line[0] in ["{", "}"]:
                _line_brace(i_line, line)
            else:
                _line_point(i_line, line)

    assert len(points) == 0, f"Braces are not paired ({fpath})"

    # set ret_dict
    frames_np = np.stack(frames)
    if frames_np.shape[0] == 1:
        frames_np = frames_np[0]
    return frames_np, info_dict


def read_toml(fpath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = toml.load(fpath)
    ret_dict: Dict[str, Any] = dict(type="unknown")
    for key in data:
        if key == "n_points":
            ret_dict[key] = int(data[key])
        elif key == "points":
            ret_dict[key] = np.asarray(data[key], dtype=np.float32)
        # other keys
        else:
            ret_dict[key] = data[key]

    # check necessary keys
    assert "n_points" in ret_dict, f"Failed to find 'n_points' ({fpath})"
    assert "points" in ret_dict or "frames" in ret_dict, f"Failed to find 'points' or 'frames'. ({fpath})"
    if "points" in ret_dict:
        assert ret_dict["n_points"] == len(
            ret_dict["points"]
        ), f"Should be {ret_dict['n_points']} points, but {len(ret_dict['points'])} are found ({fpath})"
        frames_np: np.ndarray = np.asarray(ret_dict["points"], dtype=np.float32)
        ret_dict.pop("points")
    elif "frames" in ret_dict:
        frames = []
        for i_frm, frame in enumerate(ret_dict["frames"]):
            assert "points" in frame, f"Failed to find 'points' in frames[{i_frm}] ({fpath})"
            assert ret_dict["n_points"] == len(frame["points"]), (
                f"Should be {ret_dict['n_points']} points, but {len(frame['points'])} "
                f"are found in frames[{i_frm}] ({fpath})"
            )
            frames.append(frame["points"])
        frames_np = np.asarray(frames, dtype=np.float32)
    else:
        raise NotImplementedError("Impossible!")

    return frames_np, ret_dict
