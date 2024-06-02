import math

import torch.nn as nn
import torch.nn.functional as F


class AutoPadding(nn.Module):
    __constants__ = ["method", "mode", "value"]

    def __init__(self, conv, method="same", mode="reflect", value=0):
        super().__init__()
        assert isinstance(conv, (nn.Conv1d, nn.Conv2d))
        self.conv = conv
        self.method = method
        self.mode = mode
        self.value = value

    def forward(self, x):
        x = AutoPadding._pad(
            x,
            self.conv.kernel_size,
            self.conv.stride,
            self.conv.dilation,
            self.method,
            self.mode,
            self.value,
        )
        return self.conv(x)

    @staticmethod
    def get_pad_tuple(method, padlr):
        assert padlr >= 0
        if method == "same":
            left = padlr // 2
            right = padlr - left
            return (left, right)
        elif method == "causal":
            return (padlr, 0)
        elif method == "valid":
            return (0, 0)
        else:
            raise ValueError("unknown padding mode: {}".format(method))

    @staticmethod
    def _get_pad_size(size, kernel_size, stride, dilation, method):
        padlr = (int(math.ceil(size / stride)) - 1) * stride + dilation * (kernel_size - 1) + 1 - size
        pad_tup = AutoPadding.get_pad_tuple(method, padlr)
        return pad_tup

    @staticmethod
    def _pad(inputs, kernel_size, stride, dilation, method, mode, value):
        """
        `inputs` should be in shape: (B, C, T) or (B, C, W, H).
        `kernel_size` and `stride` should be <class 'int'>, if inputs is (B, C, T).
        `padding` should be in ['same', 'valid', 'causal']
        """

        assert inputs.dim() == 3 or inputs.dim() == 4, "inputs.dim() == {}, shoule be 3 or 4".format(inputs.dim())
        assert inputs.dim() - 2 == len(kernel_size)
        assert method in ["same", "valid", "causal"]

        # pad nothing, only use valid part
        if method == "valid":
            return inputs

        pad = []
        for idx in range(len(kernel_size)):
            i = -(idx + 1)
            pad += list(AutoPadding._get_pad_size(inputs.shape[i], kernel_size[i], stride[i], dilation[i], method))
        if mode == "zero":
            assert value == 0
            mode = "constant"
        return F.pad(inputs, pad, mode, value)

    @staticmethod
    def size_after_conv(size, kernel_size, stride, dilation=1, padding_method="valid"):
        padding = AutoPadding._get_pad_size(size, kernel_size, stride, 1, method=padding_method)
        l_out = (size + sum(padding) - dilation * (kernel_size - 1) - 1) // stride + 1
        return l_out

    @staticmethod
    def size_after_deconv(size, kernel_size, stride, dilation=1):
        l_out = (size - 1) * stride + 1 + dilation * (kernel_size - 1)
        return l_out
