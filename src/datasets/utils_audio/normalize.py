import numpy as np

from src.engine.logging import get_logger

log = get_logger("audio")


def analyze_db(wav, threshold=None):
    # get maximum
    db = 20.0 * np.log10(np.maximum(np.abs(wav), 1e-10))
    max_db = db.max()
    # mask silence dynamicly
    # threshold += max(max_db, -6.0)  # at least 0.5
    if threshold is None:
        threshold = db.min()
    mask = db >= threshold
    # all silence
    if mask.sum() == 0:
        return None, None
    # get mean db
    rms = np.sqrt(np.mean(wav[mask] ** 2))
    rms_db = 20.0 * np.log10(rms)
    return rms_db, max_db


def normalize_audio(wav, target_db=-23, threshold=None, rms_db=None, max_db=None):
    if rms_db is not None:
        assert max_db is not None
    else:
        rms_db, max_db = analyze_db(wav, threshold=threshold)
    if rms_db is None:
        # all silence
        return wav
    # delta
    delta_db = target_db - rms_db
    if delta_db + max_db > 0:
        log.warn("[rms]: max db {:.2f} will > 0," "signal will be clipped".format(max_db + delta_db))
    scale_rms = np.power(10.0, delta_db / 20.0)
    rms_wav = wav * scale_rms
    return np.clip(rms_wav, -0.999, 0.999)
