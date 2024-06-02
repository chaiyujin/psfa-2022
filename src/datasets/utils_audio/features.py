import os

import numpy as np

from src.data import audio as audio_utils

from .interpolate import interpolate_features


def export_audio_features(config, audio_path, out_dir, makedirs=True, overwrite=False):
    # * Load audio
    wav, sr = audio_utils.load_th(audio_path, config.audio.sample_rate, mono=True, use_librosa=True)

    if makedirs:
        os.makedirs(out_dir, exist_ok=True)
    # * Mel: always compute, since it's fast and configs may change.
    save_path_mel = os.path.join(out_dir, "mel.npy")
    if (not os.path.exists(save_path_mel)) or overwrite:
        mel = audio_utils.feats.mel_spectrogram.compute(config, wav)  # N, L, C
        # Delta channels
        mel_with_deltas = audio_utils.feats.fns_nograd.append_delta(mel, 1)
        mel_feats = mel_with_deltas[0].numpy()
        # Save
        np.save(save_path_mel, mel_feats)
    else:
        mel_feats = np.load(save_path_mel)
    # Check shape
    assert mel_feats.shape[1] == config.audio.mel.n_mels
    # Mel hop seconds
    mel_hop = config.audio.mel.hop_size
    if isinstance(mel_hop, int):
        mel_hop = float(mel_hop) / float(config.audio.sample_rate)

    # * DeepSpeech: cached, since it's slow and configs is fixed for pretrained DeepSpeech Network.
    save_path_ds = os.path.join(out_dir, "deepspeech.npy")
    if (not os.path.exists(save_path_ds)) or overwrite:
        ds_feats = audio_utils.feats.deepspeech.compute(config, wav)[0].numpy()
        np.save(save_path_ds, ds_feats)
    else:
        ds_feats = np.load(save_path_ds)
    # Check shape
    assert ds_feats.shape[1] == 29
    assert config.audio.deepspeech.fps == 50
    # DeepSpeech hop seconds default (1/50.0)
    ds_hop = 1.0 / float(config.audio.deepspeech.fps)
    # 60 FPS version for VOCA
    save_path_ds_60fps = os.path.join(out_dir, "deepspeech_60fps.npy")
    if not os.path.exists(save_path_ds_60fps):
        ds_60fps = interpolate_features(ds_feats, config.audio.deepspeech.fps, 60)
        np.save(save_path_ds_60fps, ds_60fps)
    else:
        ds_60fps = np.load(save_path_ds_60fps)

    # fmt: off
    return dict(
        mel_feats=mel_feats, mel_hop=mel_hop,
        ds_feats=ds_feats, ds_hop=ds_hop,
        ds_60fps=ds_60fps,
    )
    # fmt: on


DS_ZEROS = np.asarray(
    [
        -3.7865527,
        -2.0535128,
        -3.7709882,
        -5.70889,
        -5.1693134,
        -2.2239838,
        -5.7553005,
        -6.0768104,
        -3.4690156,
        -3.815745,
        -10.068141,
        -4.6493406,
        -3.6702142,
        -4.5344386,
        -4.468347,
        -4.473244,
        -6.8993325,
        -11.084519,
        -6.0632415,
        -2.6136997,
        -2.7577462,
        -5.013598,
        -8.982867,
        -5.4475403,
        -7.7492123,
        -4.8355165,
        -11.728784,
        -5.0168386,
        11.526874,
    ],
    dtype=np.float32,
)
