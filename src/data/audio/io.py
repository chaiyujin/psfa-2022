import logging
import os
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio as ta

logger = logging.getLogger(__name__)


# make sure use the new interface
ta.set_audio_backend("sox_io")


def _is_video(audio_path):
    _, ext = os.path.splitext(audio_path)
    return ext.lower() in [".mp4", ".m4v", ".mkv", ".avi"]


def load_th(
    audio_path: str,
    sample_rate: int = None,
    mono: bool = True,
    normalize: bool = False,
    use_librosa: bool = True,
    device="cpu",
) -> Tuple[torch.Tensor, int]:
    if _is_video(audio_path):
        v_path = audio_path
        audio_path = os.path.splitext(audio_path)[0] + f"_sr{sample_rate}.wav"
        os.system(f"ffmpeg -loglevel panic -i {v_path} -ar {sample_rate} {audio_path} -n")

    if use_librosa:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
        if normalize:
            y = librosa.util.normalize(y)
        y = torch.tensor(y.astype(np.float32)).unsqueeze(0)
    else:
        y, sr = ta.load(audio_path)
        assert not normalize  # TODO: normalize audio with torch audio load
        # resample if necessary
        if sample_rate is not None and sample_rate != sr:
            y = ta.transforms.Resample(sr, sample_rate)(y)
            sr = sample_rate
        # mix into mono
        if y.shape[0] > 1 and mono:
            y = y.mean(dim=0, keepdim=True)
    # move to target device
    y = y.to(device)
    return y, sr


def save_th(audio_path: str, wav_tensor: torch.Tensor, sample_rate: int, makedirs: bool = False):
    """save wav signal into .wav file
    :param audio_path: The path to save.
    :param wav_tensor: torch.Tensor, in [-1, 1]
    :param sample_rate: sample rate
    :param makedirs: create directory if necessary, default is ``False``
    """
    if makedirs:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    wav_cpu = wav_tensor.detach().cpu()
    wav_cpu = torch.clamp(wav_cpu, min=-1, max=1)  # clamp into [-1, 1]
    ta.save(audio_path, wav_cpu, sample_rate)


def display_th(wav_tensor: torch.Tensor):
    # !important: only import when used, because we have to set backend
    import matplotlib.pyplot as plt

    wav_np = wav_tensor.detach().cpu().numpy()[0]
    plt.plot(wav_np)
    plt.tight_layout()
    plt.show()
