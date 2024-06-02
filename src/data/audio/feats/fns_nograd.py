import librosa
import numpy as np
import torch


def append_delta(feats: torch.Tensor, n_deltas: int, dim_len: int = 1):
    device = feats.device
    dtype = feats.dtype
    # to numpy
    assert feats.dim() == 3
    if dim_len < 0:
        dim_len += feats.dim()
    feats_np = feats.detach().cpu().numpy()  # N, C, L
    result = np.expand_dims(np.zeros_like(feats_np), -1)
    result = np.repeat(result, n_deltas + 1, axis=-1)
    for i_n, feat in enumerate(feats_np):
        result[i_n, :, :, 0] = feat
        for i_d in range(1, n_deltas + 1):
            result[i_n, :, :, i_d] = librosa.feature.delta(feat, order=i_d, axis=dim_len - 1)
    result = torch.tensor(result, dtype=dtype, device=device)
    return result
