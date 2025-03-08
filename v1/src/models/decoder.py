#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import NonBottleneck1D, Bottleneck


__all__ = ["Decoder"]


class Decoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dims=[128, 128, 128],
        n_decoder_blocks=[1, 1, 1],
        activation=nn.ReLU(inplace=True),
        type_decoder_block='nonbottleneck1d',
        encoder_decoder_fusion="add",
        upsampling_mode="bilinear",
        num_classes=37,
    ):
        super().__init__()
        assert len(out_dims) == 3, 'Decoder has 3 modules, assert output dimensions have a length of 3.'
        assert len(n_decoder_blocks) == 3, 'Decoder has 3 modules, assert number of blocks have a length of 3.'

        self.decoder_module_1 = DecoderModule(
            in_dim=in_dim,
            out_dim=out_dims[0],
            activation=activation,
            n_decoder_blocks=n_decoder_blocks[0],
            type_decoder_block=type_decoder_block,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )

        self.decoder_module_2 = DecoderModule(
            in_dim=out_dims[0],
            out_dim=out_dims[1],
            activation=activation,
            n_decoder_blocks=n_decoder_blocks[1],
            type_decoder_block=type_decoder_block,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )

        self.decoder_module_3 = DecoderModule(
            in_dim=out_dims[1],
            out_dim=out_dims[2],
            activation=activation,
            n_decoder_blocks=n_decoder_blocks[2],
            type_decoder_block=type_decoder_block,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )
        out_channels = out_dims[2]

        self.conv_out = nn.Conv2d(out_channels, num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode, channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode, channels=num_classes)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, _ = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, _ = self.decoder_module_2(out, enc_skip_down_8)
        out, _ = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        return out
    

class DecoderModule(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation=nn.ReLU(inplace=True),
        n_decoder_blocks=1,
        type_decoder_block='nonbottleneck1d',
        encoder_decoder_fusion="add",
        upsampling_mode="bilinear",
        num_classes=37,
    ):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        blocks = []
        for _ in range(n_decoder_blocks):
            if type_decoder_block == 'bottleneck':
                blocks.append(Bottleneck(out_dim, out_dim, activation=activation))
            elif type_decoder_block == 'nonbottleneck1d':
                blocks.append(NonBottleneck1D(out_dim, out_dim, activation=activation))
            else:
                raise NotImplementedError(f'Type {type_decoder_block} not supported.')
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode, channels=out_dim)

        # for pyramid supervision
        self.side_output = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == "add":
            out += encoder_features
        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        if mode == "bilinear":
            self.align_corners = False
        else:
            self.align_corners = None

        if "learned-3x3" in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == "learned-3x3":
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, padding=0)
            elif mode == "learned-3x3-zeropad":
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                            [0.0625, 0.1250, 0.0625],
                            [0.1250, 0.2500, 0.1250],
                            [0.0625, 0.1250, 0.0625],
                        ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = "nearest"
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = F.interpolate(x, size, mode=self.mode, align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x
    

if __name__ == "__main__":
    
    def count_parameters_in_millions(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params / 1_000_000 

    decoder = Decoder(
        in_dim=2048,  
        out_dims=[128, 128, 128],
        n_decoder_blocks=[1, 1, 1]
    )
    
    print(decoder)
    print(f'Total number of parameters: {count_parameters_in_millions(decoder):.2f}M')

    encoder_outs = (
        torch.randn(1, 2048, 7, 7),   # Corresponds to enc_out
        torch.randn(1, 128, 14, 14),  # Corresponds to enc_skip_down_16 --> 128 channels
        torch.randn(1, 128, 28, 28),  # Corresponds to enc_skip_down_8 --> 128 channels
        torch.randn(1, 128, 56, 56)   # Corresponds to enc_skip_down_4 --> 128 channels
    )

    outputs = decoder(encoder_outs)
    print(f'Input tensor shape: {encoder_outs[0].shape}')
    print(f'Output tensor shape: {outputs.shape}')