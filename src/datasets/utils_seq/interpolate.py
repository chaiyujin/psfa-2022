import numpy as np


def interp(x, y, a):
    return x * (1.0 - a) + y * a


def interp_seq(seqs, i, j, a):
    return seqs[i] * (1.0 - a) + seqs[j] * a


def parse_float_index(index, max_frames=None):
    ifrm = int(index)
    jfrm = int(index) + 1
    alpha = index - int(index)
    if max_frames is not None:
        ifrm = np.clip(ifrm, 0, max_frames - 1)
        jfrm = np.clip(jfrm, 0, max_frames - 1)
    return ifrm, jfrm, alpha
