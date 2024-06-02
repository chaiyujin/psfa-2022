import os
from typing import Optional, Union

import numpy as np
import torch


def compute(
    hparams, wav_tensor: torch.Tensor, wav_lengths: Optional[torch.LongTensor] = None, **kwargs
) -> torch.Tensor:
    """Compute deepspeech feature with pre-trained tensorflow model,
    not differentible.
    """

    from .handler import convert_to_deepspeech

    config = {}
    graph_fname = os.path.expanduser(hparams.audio.deepspeech.graph_pb)
    config["deepspeech_graph_fname"] = graph_fname

    sr = hparams.audio.sample_rate
    pad_val = hparams.audio.deepspeech.get("padding_value", 0)
    # prepare input dict
    seqs = dict()
    bsz = wav_tensor.size(0)
    for i in range(bsz):
        # !important: the handler only accept signal values in [-32768, 32767]
        audio = wav_tensor[i].detach().cpu().numpy() * 32768
        audio[audio > 32767] = 32767
        audio[audio < -32768] = -32768

        if wav_lengths is not None:
            audio = audio[: int(wav_lengths[i])]
        seqs[f"seq-{i:d}"] = dict(audio=audio, sample_rate=sr)

    # get results
    results = convert_to_deepspeech(dict(subject=seqs), config)["subject"]
    feats = [results[f"seq-{i:d}"]["audio"] for i in range(bsz)]
    # pad
    lengths = []
    max_len = max(x.shape[0] for x in feats)
    for i in range(len(feats)):
        lengths.append(feats[i].shape[0])
        feats[i] = np.pad(feats[i], [[0, max_len - feats[i].shape[0]], [0, 0]], "constant", constant_values=pad_val)
    # to tensor
    th_feats = torch.tensor(feats, dtype=torch.float32, device=wav_tensor.device)
    th_lengths = torch.tensor(lengths, dtype=torch.int64, device=wav_tensor.device)  # noqa: F841

    # check features size
    assert th_feats.size(0) == bsz
    assert th_feats.size(1) == max_len
    assert th_feats.size(2) == hparams.audio.deepspeech.feat_size
    return th_feats  # N, L, C

    # # N, C, L
    # th_feats = th_feats.permute(0, 2, 1).contiguous()
    # return th_feats  # , th_lengths


def naive_decode_alphabet(hparams, ds_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    from .handler import get_alphabet_list

    # get alphabet
    graph_fname = os.path.expanduser(hparams.audio.deepspeech.graph_pb)
    alphabet_fname = os.path.join(os.path.dirname(graph_fname), "alphabet.txt")
    alphabet_list = get_alphabet_list(alphabet_fname)
    alphabet_arr = np.array(alphabet_list, dtype=str)
    # get probs
    if isinstance(ds_tensor, np.ndarray):
        ds_tensor = torch.tensor(ds_tensor)
    assert len(alphabet_arr) == ds_tensor.shape[1]
    assert ds_tensor.ndim == 3
    dist = torch.distributions.categorical.Categorical(logits=ds_tensor.permute(0, 2, 1))
    index = dist.sample().detach().cpu().numpy()
    # index = torch.argmax(ds_tensor, dim=1)
    results = alphabet_arr[index]
    return results


# def draw(hparams, ds_feats: torch.Tensor, **kwargs):
#     # check input features' shape
#     assert (
#         ds_feats.dim() == 3 and ds_feats.size(1) == hparams.audio.deepspeech.feat_size
#     ), "<deepspeech.draw>: ds_feats should be in shape (N, C, L)"
#     img_list = []
#     feats_np = ds_feats.detach().cpu().numpy()
#     for i in range(feats_np.shape[0]):
#         img = saber_draw.color_mapping(feats_np[i], flip_rows=True, **kwargs)
#         img_list.append(img)
#     img_list = np.asarray(img_list, dtype=np.float32)
#     images = torch.from_numpy(img_list).to(ds_feats.device)
#     # permute into 'CHW'
#     return images.permute(0, 3, 1, 2).contiguous()
