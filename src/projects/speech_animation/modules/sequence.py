import logging

import torch.nn as nn
from omegaconf import DictConfig

from .seq_conv import SeqConv
from .seq_lstm import SeqLSTM
from .seq_xfmr import SeqXFMR

log = logging.getLogger("Sequential")


def build_sequential(
    hparams: DictConfig, ch_audio, ch_style, src_seq_frames, tgt_seq_frames, src_seq_pads
) -> nn.Module:
    log.info(f"Using {hparams.using}")
    if hparams.using == "conv":
        return SeqConv(hparams.conv, ch_audio, ch_style, src_seq_frames, tgt_seq_frames, src_seq_pads)
    elif hparams.using == "lstm":
        return SeqLSTM(hparams.lstm, ch_audio, ch_style, src_seq_frames, tgt_seq_frames, src_seq_pads)
    elif hparams.using == "xfmr":
        return SeqXFMR(hparams.xfmr, ch_audio, ch_style, src_seq_frames, tgt_seq_frames, src_seq_pads)
    else:
        raise ValueError("unknown sequential using: {}".format(hparams.using))
