import os
from typing import Dict

import cv2
import numpy as np
import torch
from torch import Tensor

from src.constants import DEBUG
from src.datasets.utils_audio import DS_ZEROS, get_feature_window
from src.datasets.utils_seq import Range

"""Augmentation:
DeepSpeech:
1. random_time_stretch
2. random_feat_scaling
MelSpectrogram:
1. random_time_stretch: append extra frames or trunk frames, then resize.
2. random_feat_scaling: scale by random sin curve on feature dim.
3. random_feat_masking: randomly mask out some.
5. random_freq_tremolo: randomly shift frequency on some temporal frame.
4. ramdom_freq_shifting: randomly shift frequency on entire window.
"""


def load_audio_features(
    config, frm_range: Range, a: float, data_dir: str, augmentation=False, using=None, **kwargs
) -> Dict[str, Tensor]:

    audio_feat: Dict[str, Tensor] = dict()
    using = config.using_audio_features if using is None else using
    aug_opts = config.random_aug_audio if augmentation else None
    if aug_opts is None:
        aug_opts = dict()

    # prepare the ts list
    ts_list = [(float(ifrm + a) / frm_range.fps) for ifrm in frm_range]

    def _get_feat_frames(key):
        if config.company_audio == "one":
            return 1
        elif config.company_audio == "win":
            return config.audio.sliding_window[key]
        else:
            raise ValueError("Unknown 'company_audio': {}".format(config.company_audio))

    def _to_3dim(feat):
        return np.expand_dims(feat, axis=-1) if feat.ndim == 2 else feat

    # * Mel
    if "mel" in using:
        assert "mel_hop" in kwargs, "[load_audio_features]: 'mel_hop' is not given!"
        mel_seq = np.load(os.path.join(data_dir, "mel.npy"))
        mel_hop = kwargs["mel_hop"]

        # * augmentation
        aug_data = _prepare_aug_data(aug_opts)

        mel_list = []
        for ts in ts_list:
            n_frames = _get_feat_frames("mel") + aug_data.get("extra_frames", 0)
            feat_org = get_feature_window(mel_seq, mel_hop, n_frames, ts, pad_mode="zero")
            if augmentation:
                feat_new = _augment(aug_data, feat_org)
                _debug_ds("mel", feat_org, feat_new)
            else:
                feat_new = feat_org
            mel_list.append(_to_3dim(feat_new))
        audio_feat["mel"] = torch.tensor(np.asarray(mel_list, dtype=np.float32))

    # * DeepSpeech
    if "ds" in using or "deepspeech" in using:
        ds_seq = np.load(os.path.join(data_dir, "deepspeech.npy"))
        ds_hop = kwargs.get("ds_hop", 1 / 50.0)
        assert ds_hop == 1 / 50.0

        # * augmentation
        aug_data = _prepare_aug_data(aug_opts)

        ds_list = []
        for ts in ts_list:
            n_frames = _get_feat_frames("deepspeech") + aug_data.get("extra_frames", 0)
            feat_org = get_feature_window(ds_seq, ds_hop, n_frames, ts, pad_mode="value", pad_value=DS_ZEROS)
            if augmentation:
                feat_new = _augment(aug_data, feat_org)
                _debug_ds("deepspeech", feat_org, feat_new)
            else:
                feat_new = feat_org
            ds_list.append(_to_3dim(feat_new))
        audio_feat["deepspeech"] = torch.tensor(np.asarray(ds_list, dtype=np.float32))

    # * DeepSpeech 60fps: no augmentation, since no such operation required in VOCA
    if "ds_60fps" in using or "deepspeech_60fps" in using:
        ds_seq = np.load(os.path.join(data_dir, "deepspeech_60fps.npy"))
        ds_hop = 1 / 60.0

        ds_list = []
        for ts in ts_list:
            n_frames = _get_feat_frames("deepspeech_60fps")
            feat_win = get_feature_window(ds_seq, ds_hop, n_frames, ts, pad_mode="zero")
            ds_list.append(_to_3dim(feat_win))
        audio_feat["deepspeech_60fps"] = torch.tensor(np.asarray(ds_list, dtype=np.float32))

    # HACK: For upper
    if "for_upper" in using:
        assert "mel_hop" in kwargs, "[load_audio_features]: 'mel_hop' is not given!"
        mel_seq = np.load(os.path.join(data_dir, "mel.npy"))
        mel_hop = kwargs["mel_hop"]

        mel_list = []
        for ts in ts_list:
            n_frames = _get_feat_frames("for_upper")
            feat = get_feature_window(mel_seq, mel_hop, n_frames, ts, pad_mode="zero", ts_is="end")  # HACK: End ts
            mel_list.append(_to_3dim(feat[..., 0]))
        audio_feat["for_upper"] = torch.tensor(np.asarray(mel_list, dtype=np.float32))

    return audio_feat


# * Augmentation
def _prepare_aug_data(aug_opts, chance=0.3):
    aug_data = dict(
        time_stretch=aug_opts.get("time_stretch", 0) if np.random.uniform() < chance else 0,
        feat_scaling=aug_opts.get("feat_scaling", 0) if np.random.uniform() < chance else 0,
    )

    if aug_data["time_stretch"] > 0:
        aug_data["extra_frames"] = np.random.randint(0, aug_data["time_stretch"] + 1) * np.random.choice([-2, 2])

    if aug_data["feat_scaling"] > 0:
        aug_data["feat_scaling"] = np.random.uniform(0, aug_data["feat_scaling"])

    return aug_data


def _augment(aug_data, feat):
    d_feat = feat.shape[1]

    # * for time_stretch
    if aug_data.get("extra_frames", 0) != 0:
        n_frames = feat.shape[0] - aug_data["extra_frames"]
        # print("extra", aug_data["extra_frames"])
        feat = cv2.resize(feat, (d_feat, n_frames), interpolation=cv2.INTER_LINEAR)

    # * for feat_scaling
    if aug_data.get("feat_scaling", 0) > 0:
        curve = (
            np.sin(
                np.linspace(0, np.pi * 2, num=d_feat) * np.random.uniform(-np.pi / 2, np.pi / 2)
                + np.random.uniform(0, np.pi)
            )
            * aug_data["feat_scaling"]
            + 1
        )[None, ...]
        if feat.ndim == 3:
            curve = curve[..., None]
        # print("scaling")
        feat *= curve

    return feat


def _debug_ds(tag, feat_org, feat_new):
    return
    if DEBUG:
        from src.utils.painter import color_mapping

        feat_org = feat_org[..., 0] if feat_org.ndim == 3 else feat_org
        feat_new = feat_new[..., 0] if feat_new.ndim == 3 else feat_new
        feat_org = np.transpose(feat_org, (1, 0))
        feat_new = np.transpose(feat_new, (1, 0))
        im_org = cv2.resize(color_mapping(feat_org), None, fx=10, fy=10)
        im_new = cv2.resize(color_mapping(feat_new), None, fx=10, fy=10)
        cv2.imshow(tag + "_org", im_org)
        cv2.imshow(tag + "_new", im_new)
        cv2.waitKey(1)
