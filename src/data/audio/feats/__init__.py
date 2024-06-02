from typing import Optional

import numpy as np
import torch

from .. import spectrogram
from . import deepspeech, fns_nograd


class mel_spectrogram(object):
    @staticmethod
    def compute(hparams, wav, **kwargs):
        return (
            spectrogram.mel_spectrogram(
                wav,
                hparams.audio.sample_rate,
                win_size=hparams.audio.mel.win_size,
                hop_size=hparams.audio.mel.hop_size,
                win_fn=hparams.audio.mel.win_fn,
                padding=hparams.audio.mel.padding,
                n_mels=hparams.audio.mel.n_mels,
                fmin=hparams.audio.mel.fmin,
                fmax=hparams.audio.mel.fmax,
                ref_db=hparams.audio.mel.ref_db,
                top_db=hparams.audio.mel.top_db,
                normalize=hparams.audio.mel.normalize,
                clip_normalized=hparams.audio.mel.clip_normalized,
                subtract_mean=hparams.audio.mel.subtract_mean,
                preemphasis=hparams.audio.mel.preemphasis,
            )
            .permute(0, 2, 1)
            .contiguous()
        )  # N, L, C


#     @staticmethod
#     def draw(hparams, feats: torch.Tensor, **kwargs):
#         # check input features' shape
#         assert (
#             feats.dim() == 3 and feats.size(1) == hparams.audio.mel.n_mels
#         ), "<mel_spectrogram.draw>: feats should be in shape (N, C, L)"
#         img_list = []
#         feats_np = feats.detach().cpu().numpy()
#         for i in range(feats_np.shape[0]):
#             # feat's L -> image's W, feat's C -> image's H
#             img = saber_draw.color_mapping(feats_np[i], flip_rows=True, **kwargs)
#             img_list.append(img)
#         img_list = np.asarray(img_list, dtype=np.float32)
#         images = torch.from_numpy(img_list).to(feats.device)
#         # permute into 'CHW'
#         return images.permute(0, 3, 1, 2).contiguous()
#

__all__ = ["deepspeech", "mel_spectrogram", "fns_nograd"]
