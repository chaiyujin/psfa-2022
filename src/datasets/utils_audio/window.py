import numpy as np


def get_feature_window(feats, hop_sec, n_frames, ts, pad_mode="zero", pad_value=None, ts_is="center"):
    if ts_is == "center":
        # get the center index and timestamp
        i_mid = n_frames // 2
        t_mid = ts + (hop_sec / 2.0) * float(n_frames % 2 == 0)
        # init the timestamp of first frame
        t_0 = t_mid - i_mid * hop_sec
    elif ts_is == "end":
        t_0 = ts - n_frames * hop_sec
    else:
        raise ValueError(f"Unknown 'ts_is'{ts_is}, should be 'center' or 'end'")

    i_src = t_0 / hop_sec
    # get interpolate factor 'a'
    i, a = int(i_src), i_src - int(i_src)
    # prepare return
    shape = list(feats.shape)
    max_frames = shape[0]
    shape[0] = n_frames

    if pad_mode == "zero":
        pad_l = np.zeros_like(feats[0])
        pad_r = np.zeros_like(feats[0])
    elif pad_mode == "reflect":
        pad_l = np.copy(feats[0])
        pad_r = np.copy(feats[-1])
    elif pad_mode == "value":
        assert pad_value is not None
        pad_l = np.copy(pad_value)
        pad_r = np.copy(pad_value)
    else:
        raise ValueError("unknown pad_mode: {}".format(pad_mode))

    ret = np.zeros(shape, dtype=feats.dtype)
    for i_ret in range(n_frames):
        frm_i = feats[i] if 0 <= i < max_frames else (pad_l if i < 0 else pad_r)
        frm_j = feats[i + 1] if 0 <= i + 1 < max_frames else (pad_l if i + 1 < 0 else pad_r)
        ret[i_ret] = frm_i * (1.0 - a) + frm_j * a
        i += 1
    return ret
