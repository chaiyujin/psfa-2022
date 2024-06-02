import torch
import torch.nn as nn

from src.engine import ops


def _build_lstm(input_size, hidden_size, num_layers, dropout):
    lstm = nn.LSTM(
        input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout
    )
    lstm = ops.init_layer(lstm, "orthogonal", gain=1.0)
    return lstm


class SeqLSTM(nn.Module):
    def __init__(self, hparams, in_channels, z_style_channels, src_seq_frames, tgt_seq_frames, src_seq_pads):
        super().__init__()

        # # concat latents and go through lstm
        latent_size = in_channels + z_style_channels
        hidden_size = hparams.hidden_channels
        self.latent_lstm = _build_lstm(latent_size, hidden_size, hparams.num_layers, hparams.dropout)
        self.out_channels = hidden_size

    def forward(self, inputs, z_style=None, **kwargs):
        x = inputs  # N, L, C
        if z_style is not None:
            assert z_style.ndim == 2 or (z_style.ndim == 3 and z_style.shape[1] == 1)
            if z_style.ndim == 2:
                z_style = z_style.unsqueeze(1)
            x = torch.cat((x, z_style.expand(-1, x.shape[1], -1)), dim=2)

        # get hc
        hc = None
        if kwargs.get("latent_lstm") is not None:
            hc_dict = kwargs.get("latent_lstm")["state"]
            assert isinstance(hc_dict, dict)
            hc = hc_dict.get("hc", None)
        # lstm
        z, hc = self.latent_lstm(x, hc)
        z = z.contiguous()
        # set hc
        if kwargs.get("latent_lstm") is not None:
            kwargs.get("latent_lstm")["state"]["hc"] = hc

        return z
