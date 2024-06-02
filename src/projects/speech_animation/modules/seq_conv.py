import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict

from src.engine.logging import get_logger
from src.engine.misc.table import Table
from src.engine.ops import parse_norm
from src.modules import AutoPadding
from src.modules.layers import Conv1d

log = get_logger("AnimNet")


class SeqConv(nn.Module):
    def __init__(self, hparams, in_channels, z_style_channels, src_seq_frames, tgt_seq_frames, src_seq_pads):
        super().__init__()
        self.in_channels = in_channels
        self.z_style_chs = z_style_channels
        self._build_encoder(hparams, encoder_norm_method=hparams.norm_method)

    @property
    def out_channels(self):
        return self._encoded_channels

    def _build_encoder(self, conv_opt, encoder_norm_method):
        hidden_channels = conv_opt.hidden_channels
        assert isinstance(hidden_channels, (list, tuple, ListConfig))

        # get kernel_size for each layer
        ksz_list = conv_opt.kernel_size
        if isinstance(ksz_list, int):
            ksz_list = [conv_opt.kernel_size for _ in hidden_channels]

        # get dilation for each layer
        dil_list = conv_opt.get("dilation", 1)
        if isinstance(dil_list, int):
            dil_list = [dil_list for _ in hidden_channels]

        # check the size is same
        assert len(hidden_channels) == len(ksz_list)
        assert len(hidden_channels) == len(dil_list)

        conv_layers = []
        self._inp_list = (self.in_channels,) + tuple(hidden_channels[:-1])
        self._out_list = tuple(hidden_channels)
        # * build each layer
        for i_layer, (inp, out, ksz, dil) in enumerate(zip(self._inp_list, self._out_list, ksz_list, dil_list)):
            if out == -1:
                out = self.in_channels
            inp += self.z_style_chs
            # * layer
            conv = Conv1d(inp, out, ksz, stride=1, padding=0, dilation=dil)
            print(conv)
            actv = nn.LeakyReLU(0.2)
            norm = parse_norm(encoder_norm_method, out)
            if conv_opt.dropout > 0:
                drop = nn.Dropout(p=conv_opt.dropout)
            else:
                drop = nn.Identity()
            # * append
            conv_layers.append(nn.Sequential(conv, actv, norm, drop))
            # * update encoded channels
            self._encoded_channels = out
        self._encoder_convs = nn.ModuleList(conv_layers)

    def forward(self, inputs, z_style=None, **kwargs):
        x = inputs.permute(0, 2, 1).contiguous()  # NLC -> NCL
        if z_style is not None:
            assert z_style.ndim == 2 or (z_style.ndim == 3 and z_style.shape[1] == 1)
            if z_style.ndim == 2:
                z_style = z_style.unsqueeze(-1)
            else:
                z_style = z_style.permute(0, 2, 1)

        # print('conv ~ encoder input:', x.shape)
        for i_layer, conv in enumerate(self._encoder_convs):
            if z_style is not None:
                x = torch.cat((x, z_style.expand(-1, -1, x.shape[2])), dim=1)
            x = conv(x)
        #     print(f'conv ~ encoder layer {i_layer}:', x.shape)
        # quit(1)

        return x.permute(0, 2, 1).contiguous()  # NCL -> NLC

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                              Pre-compute padding                                             * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @staticmethod
    def compute_padding_information(config: DictConfig):
        assert config.src_seq_frames == config.tgt_seq_frames
        conv_opt: DictConfig = config.sequential.conv
        padding_method = "same" if not conv_opt.causal else "causal"

        # * Get args
        length = config.tgt_seq_frames
        ksz_list = conv_opt.kernel_size
        dil_list = conv_opt.get("dilation", 1)
        n_layers = len(conv_opt.hidden_channels)

        # extend kernel_size into list
        stride = 1
        if isinstance(ksz_list, int):
            ksz_list = [ksz_list for _ in range(n_layers)]
        if isinstance(dil_list, int):
            dil_list = [dil_list for _ in range(n_layers)]
        assert len(ksz_list) == n_layers
        assert len(dil_list) == n_layers

        # * Reverse the conv ops and check the necessary length.
        size = length
        for i in range(n_layers):
            size = AutoPadding.size_after_deconv(size, ksz_list[-i - 1], stride, dil_list[-i - 1])
            # print("deconv", size)

        # * We got necessary size of input, compute padding
        # required_input_pads = tuple([int(np.ceil((size - length) / 2)) for _ in range(2)])
        required_input_pads = AutoPadding.get_pad_tuple(padding_method, size - length)
        required_input_size = length + sum(required_input_pads)
        # print("require", required_input_size, "with padding", required_input_pads)

        # * Step 3: Set the Wanna encoder size list to prevent AutoPadding operation, which is default.
        size = required_input_size
        for i in range(n_layers):
            size = AutoPadding.size_after_conv(size, ksz_list[i], stride, dil_list[i], padding_method="valid")
            # print(f"After Conv{i}:", size)

        # * Step 4: Overwrite config
        with open_dict(config):
            config.src_seq_pads = required_input_pads

        # fmt: off
        table = Table(alignment=("left", "middle"))
        table.add_row("animnet.src_seq_frames", config.src_seq_frames)
        table.add_row("animnet.tgt_seq_frames", config.tgt_seq_frames)
        table.add_row("animnet.src_seq_pads",   str(config.src_seq_pads))
        log.info("Pre-Computed Paddings:\n{}".format(table))
        # fmt: on
