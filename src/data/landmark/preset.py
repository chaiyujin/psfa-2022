from typing import Dict, List, Tuple

import numpy as np

# fmt: off

PRESET_LINES_OF_PARTS: Dict[str, Dict[str, List[int]]] = {
    'FW-75': {
        "CONTOUR":       list(range(0, 15)),                            # contour
        "EYEBROW-LEFT":  list(range(15, 21)) + [15],                    # eyebrow
        "EYEBROW-RIGHT": list(range(21, 27)) + [21],                    # eyebrow
        "EYELID-LEFT":   [31, 72, 32, 69, 33, 70, 34, 71, 31],          # eye
        "EYELID-RIGHT":  [27, 65, 28, 68, 29, 67, 30, 66, 27],          # eye
        "NOSE":          list(range(35, 44)),                           # nose
        "LIP-UPPER":     list(range(46, 53)) + [61, 62, 63, 46],        # upper lip
        "LIP-LOWER":     list(range(52, 58)) + [46, 58, 59, 60, 52],    # lower lip
    }
}

PRESET_LINES: Dict[str, List[int]] = {
    key: sum((indices for part, indices in parts.items()), [])
    for key, parts in PRESET_LINES_OF_PARTS.items()
}

PRESET_TRIANGLES_OF_PARTS: Dict[str, Dict[str, Tuple[Tuple[int, int, int], ...]]] = {
    'FW-75@LOWER-WO-MOUTH': {
        'LIP-LOWER': (
            (52, 53, 60),
            (53, 54, 60),
            (54, 55, 60),
            (55, 59, 60),
            (59, 55, 58),
            (55, 56, 58),
            (56, 57, 58),
            (57, 46, 58),
        ),
        'LIP-UPPER': (
            (52, 61, 51),
            (51, 61, 50),
            (50, 61, 49),
            (49, 61, 62),
            (62, 48, 49),
            (48, 62, 63),
            (63, 47, 48),
            (63, 46, 47),
        ),
        'CHIN-RIGHT': (
            (7,  55, 54),
            (7,  54, 8 ),
            (8,  54, 53),
            (8,  53, 9 ),
            (9,  53, 52),
            (9,  52, 10),
            (10, 52, 11),
        ),
        'CHIN-LEFT': (
            (7,  56, 55),
            (7,  6,  56),
            (6,  57, 56),
            (6,  5,  57),
            (5,  46, 57),
            (5,  4,  46),
            (4,  3,  46),
        ),
        'PHILTRUM': (
            # RIGHT
            (39, 50, 49),
            (39, 40, 50),
            (40, 51, 50),
            (40, 52, 51),
            (40, 11, 52),
            # LEFT
            (39, 49, 48),
            (39, 48, 38),
            (38, 48, 47),
            (38, 47, 46),
            (38, 46, 3 ),
        )
    },
    "LF-30": {
        'LIP-LOWER': (
            (18, 19, 26),
            (19, 20, 26),
            (20, 21, 26),
            (21, 25, 26),
            (25, 21, 24),
            (21, 22, 24),
            (22, 23, 24),
            (23, 12, 24),
        ),
        'LIP-UPPER': (
            (18, 27, 17),
            (17, 27, 16),
            (16, 27, 15),
            (15, 27, 28),
            (28, 14, 15),
            (14, 28, 29),
            (29, 13, 14),
            (29, 12, 13),
        ),
        'CHIN-RIGHT': (
            ( 4, 21, 20),
            ( 4, 20,  5),
            ( 5, 20, 19),
            ( 5, 19,  6),
            ( 6, 19, 18),
            ( 6, 18,  7),
            ( 7, 18,  8),
        ),
        'CHIN-LEFT': (
            ( 4, 22, 21),
            ( 4,  3, 22),
            ( 3, 23, 22),
            ( 3,  2, 23),
            ( 2, 12, 23),
            ( 2,  1, 12),
            ( 1,  0, 12),
        ),
        'PHILTRUM': (
            (10, 16, 15),
            (10, 11, 16),
            (11, 17, 16),
            (11, 18, 17),
            (11,  8, 18),
            (10, 15, 14),
            (10, 14,  9),
            ( 9, 14, 13),
            ( 9, 13, 12),
            ( 9, 12,  0),
        ),
    }
}

PRESET_TRIANGLES: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    key: sum((tris for part, tris in parts.items()), ())
    for key, parts in PRESET_TRIANGLES_OF_PARTS.items()
}

PRESET_MAPPING: Dict[str, Dict[int, int]] = {
    'FW-75 -> LF-30': {
        3:  0,  4:  1,  5:  2,  6:  3,  7:  4,
        8:  5,  9:  6,  10: 7,  11: 8,  38: 9,
        39: 10, 40: 11, 46: 12, 47: 13, 48: 14,
        49: 15, 50: 16, 51: 17, 52: 18, 53: 19,
        54: 20, 55: 21, 56: 22, 57: 23, 58: 24,
        59: 25, 60: 26, 61: 27, 62: 28, 63: 29
    }
}

PRESET_TEMPLATE: Dict[str, np.ndarray] = {
    'FW-75': np.asarray([
        [ 0.79624420, -0.44739336],
        [ 0.78122073, -0.22748810],
        [ 0.74366194, -0.00758290],
        [ 0.69107991,  0.22748822],
        [ 0.59342724,  0.43981045],
        [ 0.44319254,  0.62938398],
        [ 0.24037558,  0.75829393],
        [ 0.00000000,  0.79620856],
        [-0.24037558,  0.75829393],
        [-0.44319248,  0.62938398],
        [-0.59342730,  0.43981045],
        [-0.69107980,  0.22748822],
        [-0.74366200, -0.00758290],
        [-0.78122067, -0.22748810],
        [-0.79624414, -0.44739336],
        [-0.62347424, -0.62938386],
        [-0.49577469, -0.77345967],
        [-0.30798125, -0.79620850],
        [-0.14272302, -0.72037911],
        [-0.30798125, -0.70521325],
        [-0.47323942, -0.68246442],
        [ 0.62347418, -0.62938386],
        [ 0.49577469, -0.77345967],
        [ 0.30798119, -0.79620850],
        [ 0.14272302, -0.72037911],
        [ 0.30798119, -0.70521325],
        [ 0.47323948, -0.68246442],
        [ 0.51830989, -0.55355448],
        [ 0.36807507, -0.62938386],
        [ 0.20281690, -0.55355448],
        [ 0.36056334, -0.51563978],
        [-0.51830989, -0.55355448],
        [-0.36807513, -0.62938386],
        [-0.20281690, -0.55355448],
        [-0.36056334, -0.51563978],
        [ 0.11267608, -0.55355448],
        [ 0.14272302, -0.35639805],
        [ 0.25539905, -0.15165871],
        [ 0.20281690, -0.06824642],
        [ 0.00000000, -0.05308050],
        [-0.20281690, -0.06824642],
        [-0.25539905, -0.15165871],
        [-0.14272302, -0.35639805],
        [-0.11267602, -0.55355448],
        [ 0.12769943, -0.11374402],
        [-0.12769955, -0.11374402],
        [ 0.30798119,  0.25023693],
        [ 0.19530517,  0.14407581],
        [ 0.07511741,  0.10616118],
        [ 0.00000000,  0.12132710],
        [-0.07511741,  0.10616118],
        [-0.19530517,  0.14407581],
        [-0.30798125,  0.25023693],
        [-0.22535211,  0.34123224],
        [-0.11267602,  0.38672990],
        [ 0.00000000,  0.40947872],
        [ 0.11267608,  0.38672990],
        [ 0.22535211,  0.34123224],
        [ 0.12769943,  0.25781995],
        [ 0.00000000,  0.26540285],
        [-0.12769955,  0.25781995],
        [-0.12769955,  0.20473939],
        [ 0.00000000,  0.19715637],
        [ 0.12769943,  0.20473939],
        [ 0.00000000, -0.19715637],
        [ 0.45821589, -0.59905213],
        [ 0.44319254, -0.52322268],
        [ 0.28544599, -0.52322268],
        [ 0.27042252, -0.62180090],
        [-0.27042252, -0.62180090],
        [-0.28544599, -0.52322268],
        [-0.44319248, -0.52322268],
        [-0.45821601, -0.59905213],
        [ 0.35305172, -0.57630330],
        [-0.35305166, -0.57630330],
    ], dtype=np.float32),



    'LF-30': np.asarray([
        [ 0.82929581, -0.09069157],
        [ 0.71211255,  0.16409512],
        [ 0.53183091,  0.39158335],
        [ 0.28845057,  0.54627532],
        [ 0.00000000,  0.59177285],
        [-0.28845084,  0.54627532],
        [-0.53183115,  0.39158335],
        [-0.71211290,  0.16409512],
        [-0.82929593, -0.09069157],
        [ 0.24338014, -0.44557315],
        [ 0.00000000, -0.44557315],
        [-0.24338044, -0.44557315],
        [ 0.36957729, -0.06339312],
        [ 0.23436607, -0.19078647],
        [ 0.09014075, -0.23628403],
        [ 0.00000000, -0.21808492],
        [-0.09014104, -0.23628403],
        [-0.23436636, -0.19078647],
        [-0.36957765, -0.06339312],
        [-0.27042270,  0.04580126],
        [-0.13521138,  0.10039845],
        [ 0.00000000,  0.12769705],
        [ 0.13521117,  0.10039845],
        [ 0.27042240,  0.04580126],
        [ 0.15323919, -0.03229349],
        [ 0.00000000, -0.03229349],
        [-0.15323961, -0.03229349],
        [-0.15323961, -0.12599017],
        [ 0.00000000, -0.12599017],
        [ 0.15323919, -0.12599017],
    ], dtype=np.float32)
}