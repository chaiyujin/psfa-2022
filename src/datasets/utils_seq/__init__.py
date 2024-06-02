import numpy as np

from src.constants import DEFAULT_AVOFFSET_MS

from .coordinates import get_frm_range_from_sub_range, process_data_dicts
from .interpolate import interp, interp_seq, parse_float_index
from .types import FrmCoord, Range, SeqCoord


def avoffset_to_frame_delta(avoffset):
    if avoffset is None:
        return 0
    return avoffset * -1


def avoffset_from_ms(offset_ms, fps):
    offset_ms -= DEFAULT_AVOFFSET_MS
    return int(np.round(fps * offset_ms / 1000.0))
