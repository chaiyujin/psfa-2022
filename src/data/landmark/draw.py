from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .io import denormalize
from .preset import PRESET_LINES_OF_PARTS, PRESET_TRIANGLES_OF_PARTS


def draw_landmarks(
    # basic arguments
    cv_canvas: np.ndarray,
    lmks: np.ndarray,
    radius: int = None,
    color: Tuple[int, int, int] = (255, 255, 255),
    # supporting arguments
    is_lmks_normalized: bool = False,
    draw_preset_lines: str = "none",
    draw_preset_triangles: str = "none",
    draw_preset_fill: bool = False,
    draw_preset_thickness: int = 1,
    draw_preset_color: Tuple[int, int, int] = (255, 255, 255),
    # copy or not
    copy: bool = True,
) -> np.ndarray:

    assert cv_canvas.ndim == 3 and cv_canvas.shape[2] in [1, 3, 4]

    if copy:
        new_canvas = cv_canvas.copy()
    else:
        new_canvas = cv_canvas

    # unnormalize landmarks
    if is_lmks_normalized:
        lmks = denormalize(lmks, (cv_canvas.shape[1], cv_canvas.shape[0]))

    if isinstance(lmks, np.ndarray):
        lmks = np.round(lmks)
    elif torch.is_tensor(lmks):
        lmks = torch.round(lmks)

    # draw lines
    if draw_preset_lines != "none" and draw_preset_lines != "NONE":
        assert (
            draw_preset_lines.upper() in PRESET_LINES_OF_PARTS
        ), "Given 'draw_preset_lines' {} is invalid, should be in {}".format(
            draw_preset_lines, list(PRESET_LINES_OF_PARTS.keys())
        )
        for part, pt_id_list in PRESET_LINES_OF_PARTS[draw_preset_lines.upper()].items():
            for i in range(len(pt_id_list) - 1):
                p0 = tuple(int(x) for x in lmks[pt_id_list[i]])
                p1 = tuple(int(x) for x in lmks[pt_id_list[i + 1]])
                cv2.line(new_canvas, p0, p1, draw_preset_color, draw_preset_thickness)
    # draw triangles
    if draw_preset_triangles != "none" and draw_preset_triangles != "NONE":
        assert (
            draw_preset_triangles.upper() in PRESET_TRIANGLES_OF_PARTS
        ), "Given 'draw_preset_triangles' {} is invalid, should be in {}".format(
            draw_preset_triangles, list(PRESET_TRIANGLES_OF_PARTS.keys())
        )
        for part, tri_list in PRESET_TRIANGLES_OF_PARTS[draw_preset_triangles.upper()].items():
            for tri in tri_list:
                pts = np.asarray(
                    [
                        [int(x) for x in lmks[tri[0]]],
                        [int(x) for x in lmks[tri[1]]],
                        [int(x) for x in lmks[tri[2]]],
                    ],
                    dtype=np.int32,
                )
                # draw
                if draw_preset_fill:
                    cv2.fillPoly(new_canvas, [pts], draw_preset_color)
                else:
                    cv2.polylines(new_canvas, [pts], True, draw_preset_color, draw_preset_thickness)

    # draw points
    if radius is None:
        radius = max(1, int(np.round(2.0 * new_canvas.shape[0] / 512.0)))
    for pt in lmks:
        if pt[0] < 0 or pt[0] > new_canvas.shape[1] or pt[1] < 0 or pt[1] > new_canvas.shape[0]:
            continue
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(new_canvas, center, radius, color, thickness=-1)

    return new_canvas
