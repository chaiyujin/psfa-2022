import logging
import os

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)


_git_path = os.path.join(os.path.dirname(__file__), "rnnoise/")
_git_url = "https://github.com/chaiyujin/rnnoise.git"
_bin_path = os.path.join(_git_path, "examples/rnnoise_demo")


def _make():
    cur_path = os.getcwd()
    # begin to make
    os.chdir(_git_path)
    os.system("./autogen.sh && ./configure && make")
    # exit make
    os.chdir(cur_path)


def run_rnnoise_demo(wav, sr, specific_key=""):
    """'specific_key' is used to handle multi-thread"""

    is_tensor = False
    device = None
    dtype = None
    is_batch = False

    if torch.is_tensor(wav):
        device = wav.device
        dtype = wav.dtype
        is_tensor = True
        assert not wav.requires_grad
        wav = wav.detach().cpu().numpy()
    if wav.ndim > 2:
        raise ValueError(f"given wav has shape: {wav.shape}, should be dim 1 or 2(batched)")
    if wav.ndim == 2:
        is_batch = True
    else:  # ndim == 1
        is_batch = False
        wav = wav[None, ...]

    results = []

    tmp_noise = os.path.join(_git_path, f"tmp_input_{specific_key}.pcm")
    tmp_after = os.path.join(_git_path, f"tmp_output_{specific_key}.pcm")

    for signal in wav:
        # resample
        resampled = librosa.resample(signal, sr, 48000)
        # -> int16
        resampled = resampled * 32768.0
        resampled[resampled > 32767] = 32767
        resampled[resampled < -32768] = -32768
        resampled = resampled.astype(np.int16)
        # pad
        resampled = np.pad(resampled, [[0, 1024]], "constant")
        # run
        resampled.tofile(tmp_noise)
        ret = os.system("{} {} {}".format(_bin_path, tmp_noise, tmp_after))
        if ret != 0:
            raise RuntimeError("Error when denoise.")
        denoised = np.fromfile(tmp_after, dtype=np.int16)
        # adjust length
        real_length = len(resampled) - 1024
        denoised = denoised[:real_length]
        if len(denoised) < real_length:
            denoised = np.pad(denoised, [[0, real_length - len(denoised)]], "constant")
        # print(len(resampled))
        # print(len(denoised), real_length)
        # import matplotlib.pyplot as plt
        # plt.plot(resampled[351000*3:real_length], label="original")
        # plt.plot(denoised[351000*3:], label="denoised")
        # plt.legend()
        # plt.show()
        # quit()
        denoised = denoised.astype(np.float32) / 32768.0
        denoised = librosa.resample(denoised, 48000, sr)
        os.remove(tmp_noise)
        os.remove(tmp_after)
        results.append(denoised)

    if not is_batch:
        results = results[0]
    else:
        results = np.asarray(results)

    if is_tensor:
        results = torch.tensor(results, dtype=dtype, device=device)

    return results


# * check and auto clone, make
if not os.path.exists(_git_path):
    logger.warning("Failed to find 'rnnoise', git clone.")
    os.system("git clone {} {}".format(_git_url, _git_path))
if not os.path.exists(_bin_path):
    _make()
