import librosa
import numpy as np
import webrtcvad
from scipy.interpolate import interp1d


def resize(signal, target_length, kind="linear"):
    old_x = np.linspace(0, 1, num=len(signal), endpoint=True)
    new_x = np.linspace(0, 1, num=target_length, endpoint=True)
    func = interp1d(old_x, signal, kind=kind)
    return np.asarray([func(x) for x in new_x])


def detect_speech(signal, sample_rate, pad_mode="constant", smooth_ms=None, vad_mode=3):
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    # init a vad
    assert 0 <= vad_mode <= 3
    vad = webrtcvad.Vad(vad_mode)  # 0~3, 3 is most aggresive mode

    # store original length
    original_length = len(signal)

    # pad and resample
    sr = 16000
    win_len = int(0.02 * sr)
    hop_len = int(0.02 * sr)

    if sample_rate != sr:
        signal = librosa.resample(signal, sample_rate, sr)

    # pad
    to_pad = (win_len - hop_len) // 2
    signal = np.pad(signal, [to_pad, to_pad], pad_mode)

    # framing
    frames = [np.copy(signal[l : l + win_len]) for l in range(0, len(signal) - win_len, hop_len)]

    # detect
    is_speech = []
    for frame in frames:
        frame = (frame * 32768.0).astype(np.int16)
        is_speech.append(vad.is_speech(frame.tobytes(), sr))
    is_speech = np.asarray(is_speech, np.uint8)

    # smoothing
    if smooth_ms is not None:
        threshold = smooth_ms / 2.5  # (smooth_ms * sr / 1000) / hop_len
        i, last, ret = 0, 0, []
        while i < len(is_speech):
            j = i
            while j < len(is_speech) and is_speech[i] == is_speech[j]:
                j += 1
            cur = is_speech[i]
            if j - i < threshold:
                cur = last
            last = cur
            for k in range(i, j):
                ret.append(cur)
            i = j
        ret = np.asarray(ret, np.uint8)
    else:
        ret = is_speech

    # expand into original length
    ret = np.repeat(ret, repeats=hop_len)
    if len(signal) > len(ret):
        ret = np.pad(ret, [[0, len(signal) - len(ret)]], "constant", constant_values=ret[-1])

    if sample_rate != sr:
        ret = np.asarray(ret, np.float32)
        ret = resize(ret, original_length)

    return ret.astype(np.uint8)


def to_pairs(vad):
    pairs = []
    i = 0
    while i < len(vad):
        while i < len(vad) and vad[i] == 0:
            i += 1
        if i >= len(vad):
            break
        j = i + 1
        while j < len(vad) and vad[j] == 1:
            j += 1
        pairs.append((i, j))
        i = j
    return pairs


def from_pairs(pairs, length):
    vad = np.zeros((length), np.uint8)
    for l, r in pairs:
        vad[l:r] = 1
    return vad
