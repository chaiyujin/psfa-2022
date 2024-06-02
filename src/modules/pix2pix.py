from typing import Tuple, Union

import torch.nn as nn


class Pix2Pix(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_local_enhancers=1,
        n_local_residual_blocks=3,
        n_global_downsampling=3,
        n_global_residual_blocks=3,
        norm_method="instance",
        padding_type="reflect",
        activation="relu",
    ):
        super().__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        self.global_generator = GlobalGenerator.build_generator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf_global,
            n_downsampling=n_global_downsampling,
            n_blocks=n_global_residual_blocks,
            norm_method=norm_method,
            padding_type=padding_type,
            activation=activation,
            final_conv=False,  # get rid of final convolution layers
        )

        ###### local enhancer layers #####
        self.local_dwsamplers = nn.ModuleList()
        self.local_upsamplers = nn.ModuleList()
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [
                _parse_padding(padding_type, 3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                _parse_norm(norm_method, ngf_global),
                _parse_activation(activation),
                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                _parse_norm(norm_method, ngf_global * 2),
                _parse_activation(activation),
            ]
            ### residual blocks
            model_upsample = []
            for _ in range(n_local_residual_blocks):
                dim = ngf_global * 2
                model_upsample += [
                    ResnetBlock(dim, padding_type=padding_type, norm_method=norm_method, activation=activation)
                ]
            ### upsample to prevent checkerboard effect
            model_upsample += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                _parse_padding(padding_type, 1),
                nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=0),
                _parse_norm(norm_method, ngf_global),
                _parse_activation(activation),
            ]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    _parse_padding(padding_type, 3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh(),
                ]

            ### append
            self.local_dwsamplers.append(nn.Sequential(*model_downsample))
            self.local_upsamplers.append(nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for _ in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.global_generator(input_downsampled[-1])

        ### build up one layer at a time
        for n in range(1, self.n_local_enhancers + 1):
            model_dwsample = self.local_dwsamplers[n - 1]
            model_upsample = self.local_upsamplers[n - 1]
            input_i = input_downsampled[self.n_local_enhancers - n]
            output_prev = model_upsample(model_dwsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=3,
        norm_method="instance",
        padding_type="reflect",
        activation="relu",
    ):
        super().__init__()
        self.model = self.build_generator(
            input_nc, output_nc, ngf, n_downsampling, n_blocks, norm_method, padding_type, activation, final_conv=True
        )

    def forward(self, input):
        return self.model(input)

    @staticmethod
    def build_generator(
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=3,
        norm_method="instance",
        padding_type="reflect",
        activation="relu",
        final_conv=True,
    ):
        assert n_blocks >= 0
        # input
        model = [
            _parse_padding(padding_type, 3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            _parse_norm(norm_method, ngf),
            _parse_activation(activation),
        ]

        # downsample
        for i in range(n_downsampling):
            mult = 2**i
            nc_inp = ngf * mult
            nc_out = ngf * mult * 2
            model += [
                nn.Conv2d(nc_inp, nc_out, kernel_size=3, stride=2, padding=1),
                _parse_norm(norm_method, nc_out),
                _parse_activation(activation),
            ]

        # resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_method=norm_method)
            ]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            nc_inp = ngf * mult
            nc_out = ngf * mult // 2
            model += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                _parse_padding(padding_type, 1),
                nn.Conv2d(nc_inp, nc_out, kernel_size=3, stride=1, padding=0),
                _parse_norm(norm_method, nc_out),
                _parse_activation(activation),
            ]

        # final conv to output image
        if final_conv:
            model += [_parse_padding(padding_type, 3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        return nn.Sequential(*model)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_method, activation="relu", use_dropout=False):
        super(ResnetBlock, self).__init__()

        conv_block = [
            _parse_padding(padding_type, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            _parse_norm(norm_method, dim),
            _parse_activation(activation),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            _parse_padding(padding_type, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            _parse_norm(norm_method, dim),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def _parse_padding(padding_type, padding: Union[int, Tuple[int, int]]):
    if padding_type == "reflect":
        return nn.ReflectionPad2d(padding)
    elif padding_type == "replicate":
        return nn.ReplicationPad2d(padding)
    elif padding_type == "zero":
        return nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError("padding [%s] is not implemented" % padding_type)


def _parse_activation(activation):
    if activation == "relu":
        return nn.ReLU(True)
    else:
        raise NotImplementedError()


def _parse_norm(norm, dim):
    norm = norm.lower()
    if norm in ["instance", "instance_norm", "in"]:
        return nn.InstanceNorm2d(dim)
    elif norm in ["batch", "batch_norm", "bn"]:
        return nn.BatchNorm2d(dim)
    elif norm in [None, "none"]:
        return nn.Identity()
    else:
        raise NotImplementedError("Unknown norm method: {}".format(norm))
