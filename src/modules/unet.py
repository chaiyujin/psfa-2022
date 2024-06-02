import functools
import logging
import re

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class UNetSkipConnection(nn.Module):
    __valid_upsample_methods__ = ["conv-transpose", "bilinear&conv", "nearest&conv", "none@conv"]
    __valid_downsample_methods__ = ["conv", "none@dilated-conv"]

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        last_activation=nn.Tanh,
        use_dropout=False,
        dropout=0.5,
        use_norm=True,
        norm_layer=nn.BatchNorm2d,
        upsample_method="conv-transpose",
        downsample_method="conv",
        dilation=1,
    ):
        super().__init__()
        # analyze arguments
        assert not (outermost and innermost)
        self.outermost = outermost
        self.innermost = innermost
        # get input_nc
        if input_nc is None:
            input_nc = outer_nc
        # auto use_bias
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # build downsample layers
        downconv = self.build_downsample_layer(downsample_method, input_nc, inner_nc, dilation=dilation, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if use_norm else None

        # build upsample layers
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc) if use_norm else None

        # build block
        if outermost:
            assert submodule is not None
            upconv = self.build_upsample_layer(upsample_method, inner_nc * 2, outer_nc, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv, last_activation()]
            model = down + [submodule] + up
        elif innermost:
            assert submodule is None
            upconv = self.build_upsample_layer(upsample_method, inner_nc, outer_nc, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm] if use_norm else [uprelu, upconv]
            if use_dropout:
                model = down + [nn.Dropout(dropout)] + up
            else:
                model = down + up
        else:
            upconv = self.build_upsample_layer(upsample_method, inner_nc * 2, outer_nc, bias=use_bias)
            down = [downrelu, downconv, downnorm] if use_norm else [downrelu, downconv]
            up = [uprelu, upconv, upnorm] if use_norm else [uprelu, upconv]
            # down   = [downrelu, downconv]  # compatibale with legacy code (from Neural Voice Puppery), should be bug
            if use_dropout:
                model = down + [nn.Dropout(dropout)] + [submodule] + up
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # tag = ""
        # if self.outermost:
        #     tag = "outermost"
        # if self.innermost:
        #     tag = "innermost"
        # print(x.shape, tag)
        if self.outermost:
            y = self.model(x)
        else:
            y = torch.cat([x, self.model(x)], 1)
        # print(y.shape, tag)
        return y

    @classmethod
    def build_downsample_layer(cls, method, input_nc, output_nc, dilation, bias):
        if method == "conv":
            return nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=bias)
        elif method == "none@dilated-conv":
            # print(dilation)
            return nn.Conv2d(
                input_nc,
                output_nc,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=1 * dilation,
                bias=bias,
            )
        else:
            raise ValueError("Unknown downsample_method: {}".format(method))

    @classmethod
    def build_upsample_layer(cls, method, input_nc, output_nc, bias):
        if method == "conv-transpose":
            return nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=bias)
        elif method == "bilinear&conv":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=bias),
            )
        elif method == "nearest&conv":
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=bias),
            )
        elif method == "none@conv":
            return nn.Conv2d(
                input_nc,
                output_nc,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding=1,
                bias=bias,
            )
        else:
            raise ValueError("Unknown upsample_method: {}".format(method))


def _parse_norm(norm_type="instance"):
    norm_type = norm_type.lower()
    if norm_type in ["batch", "bn"]:
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type in ["instance", "in"]:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type in ["none", "no"]:
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class UNetRenderer(nn.Module):
    def __init__(
        self,
        renderer,
        input_nc,
        output_nc,
        ngf=64,
        use_dropout=False,
        dropout=0.5,
        use_norm=True,
        norm_method="instance",
        last_activation=nn.Tanh,
    ):
        super(UNetRenderer, self).__init__()
        # parse num_downs
        match = re.match(r"^_*(\d+)_*((BU|DC|NU)?)$", renderer.replace("UNET", "").replace("level", ""))
        if match is None:
            raise ValueError("Unknown renderer: {}".format(renderer))
        num_downs = int(match.group(1))
        unet_type = match.group(2)

        kwargs = dict(use_norm=use_norm, norm_layer=_parse_norm(norm_method), last_activation=last_activation)
        if unet_type == "BU":
            kwargs["upsample_method"] = "bilinear&conv"
            kwargs["downsample_method"] = "conv"
        elif unet_type == "NU":
            kwargs["upsample_method"] = "nearest&conv"
            kwargs["downsample_method"] = "conv"
        elif unet_type == "DC":
            kwargs["upsample_method"] = "none@conv"
            kwargs["downsample_method"] = "none@dilated-conv"
            kwargs["dilation"] = 1
        elif unet_type == "":
            kwargs["upsample_method"] = "conv-transpose"
            kwargs["downsample_method"] = "conv"
        else:
            raise ValueError("Unknown renderer: {}".format(renderer))

        log.info(
            ">> UNet renderer: {}, Dropout: {} {}, Normalization: {} {}".format(
                renderer, use_dropout, dropout, use_norm, norm_method
            )
        )

        def _build_block(*block_args, **block_kwargs):
            block = UNetSkipConnection(*block_args, **block_kwargs)
            if "dilation" in kwargs:
                kwargs["dilation"] *= 2
            return block

        if num_downs >= 5:
            unet_block = _build_block(ngf * 8, ngf * 8, input_nc=None, submodule=None, **kwargs, innermost=True)
            for i in range(num_downs - 5):
                unet_block = _build_block(
                    ngf * 8,
                    ngf * 8,
                    input_nc=None,
                    submodule=unet_block,
                    **kwargs,
                    use_dropout=use_dropout,
                    dropout=dropout
                )
            unet_block = _build_block(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, **kwargs)
            unet_block = _build_block(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, **kwargs)
            unet_block = _build_block(ngf * 1, ngf * 2, input_nc=None, submodule=unet_block, **kwargs)
            unet_block = _build_block(
                output_nc, ngf * 1, input_nc=input_nc, submodule=unet_block, **kwargs, outermost=True
            )
        elif num_downs == 3:
            unet_block = _build_block(ngf * 2, ngf * 8, input_nc=None, submodule=None, **kwargs, innermost=True)
            unet_block = _build_block(ngf * 1, ngf * 2, input_nc=None, submodule=unet_block, **kwargs)
            unet_block = _build_block(
                output_nc, ngf * 1, input_nc=input_nc, submodule=unet_block, **kwargs, outermost=True
            )
        else:
            raise ValueError("Dont support {} downs!".format(num_downs))

        self.model = unet_block

    def forward(self, features, background=None):
        if background is not None:
            unet_input = torch.cat([features, background], 1)
        else:
            unet_input = features
        return self.model(unet_input)


if __name__ == "__main__":
    UNetRenderer("UNET_8_level_BU", 3, 3)
    UNetRenderer("UNET_3_level_DC", 3, 3)
    UNetRenderer("UNET_5_level", 3, 3)
