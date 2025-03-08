#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet18, ResNet34, ResNet50, BasicBlock
from .neck import PyramidPoolingModule, AdaptivePyramidPoolingModule, SqueezeAndExcitation
from .decoder import Decoder


class OWSNetwork(nn.Module):
    def __init__(
        self,
        height=480,
        width=640,
        num_classes=37,
        # encoder
        encoder="resnet50", # resnet18, resnet34, resnet50
        encoder_block= BasicBlock, # BasicBlock or NonBottleneck1D for resnet18/resnet34
        input_channels=3,
        activation="relu",
        pretrained_on_imagenet=False,
        pretrained_dir=None,
        # decoder
        channels_decoder=[512, 256, 128], # default: [128, 128, 128]
        encoder_decoder_fusion="add",
        n_decoder_blocks=[1, 1, 1], # default: [1, 1, 1]
        type_decoder_block='nonbottleneck1d', # nonbottleneck1d or bottleneck
        # neck: context
        neck="ppm", # ppm, appm
        upsampling="bilinear",
        # neck: SE
        use_se=True,
        reduction=16,
    ):
        super(OWSNetwork, self).__init__()

        # Activation function in encoder, decoder, neck
        if activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ["swish", "silu"]:
            self.activation = torch.nn.SiLU(inplace=True)
        elif activation.lower() == "hswish":
            self.activation = torch.nn.Hardswish(inplace=True)
        else:
            raise NotImplementedError(f'Activation {activation} not supported.')

        #######################################################
        ####################### Encoder #######################
        #######################################################
        
        if encoder == "resnet18":
            self.encoder = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=False,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels,
            )
        elif encoder == "resnet34":
            self.encoder = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=input_channels,
            )
        elif encoder == "resnet50":
            self.encoder = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=input_channels,
            )
        else:
            raise NotImplementedError(f'Encoder {encoder} is not supported.')
            
        self.channels_decoder_in = self.encoder.down_32_channels_out
        
        #######################################################
        ######################## Neck #########################
        #######################################################

        self.se_layer0 = SqueezeAndExcitation(self.encoder.down_2_channels_out, reduction) if use_se else nn.Identity()
        self.se_layer1 = SqueezeAndExcitation(self.encoder.down_4_channels_out, reduction) if use_se else nn.Identity()
        self.se_layer2 = SqueezeAndExcitation(self.encoder.down_8_channels_out, reduction) if use_se else nn.Identity()
        self.se_layer3 = SqueezeAndExcitation(self.encoder.down_16_channels_out, reduction) if use_se else nn.Identity()
        self.se_layer4 = SqueezeAndExcitation(self.encoder.down_32_channels_out, reduction) if use_se else nn.Identity()

        if encoder_decoder_fusion == "add":
            self.skip_layer1 = self._create_skip_layer(self.encoder.down_4_channels_out, channels_decoder[2])
            self.skip_layer2 = self._create_skip_layer(self.encoder.down_8_channels_out, channels_decoder[1])
            self.skip_layer3 = self._create_skip_layer(self.encoder.down_16_channels_out, channels_decoder[0])
            
        #######################################################
        #################### Neck: context ####################
        #######################################################
        
        if "learned-3x3" in upsampling:
            print("Notice: for the context module the learned upsampling is not possible. Using nearest neighbor.")
            upsampling_context_module = "nearest"
        else:
            upsampling_context_module = upsampling
            
        if neck == "ppm":
            self.neck = PyramidPoolingModule(
                self.channels_decoder_in,
                channels_decoder[0],
                bins=(1, 2, 3, 6),
                upsampling_mode=upsampling_context_module
            )
        elif neck == "appm":
            self.neck = AdaptivePyramidPoolingModule(
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),      
                bins=(1, 2, 4, 8),

                #bins=(1, 2, 3, 6),
                upsampling_mode=upsampling_context_module
            )
        else:
            self.neck = nn.Identity()
        
        channels_after_context_module = channels_decoder[0]

        #######################################################
        ####################### Decoder #######################
        #######################################################
        
        self.decoder_ss = Decoder(
            in_dim=channels_after_context_module,
            out_dims=channels_decoder,
            activation=self.activation,
            n_decoder_blocks=n_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes,
            type_decoder_block=type_decoder_block,
        )
        
        self.decoder_ow = Decoder(
            in_dim=channels_after_context_module,
            out_dims=channels_decoder,
            activation=self.activation,
            n_decoder_blocks=n_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=19,
            type_decoder_block=type_decoder_block,
        )

    def _create_skip_layer(self, channels_in, channels_out):
        layers = []
        if channels_in != channels_out:
            layers = [
                nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)
    
    def forward(self, image):
        # encoder initial block
        out = self.encoder.forward_first_conv(image)
        out = self.se_layer0(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        # encoder block 1
        out = self.encoder.forward_layer1(out)
        out = self.se_layer1(out)
        skip1 = self.skip_layer1(out)

        # encoder block 2
        out = self.encoder.forward_layer2(out)
        out = self.se_layer2(out)
        skip2 = self.skip_layer2(out)

        # encoder block 3
        out = self.encoder.forward_layer3(out)
        out = self.se_layer3(out)
        skip3 = self.skip_layer3(out)

        # encoder block 4
        out = self.encoder.forward_layer4(out)
        out = self.se_layer4(out)

        # neck: context
        out = self.neck(out)
        
        # concat outputs
        out = [out, skip3, skip2, skip1]
        
        return self.decoder_ss(enc_outs=out), self.decoder_ow(enc_outs=out)


if __name__ == "__main__":
    
    def count_parameters_in_millions(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params / 1_000_000 
    
    model = OWSNetwork()
    print(f'Total number of parameters: {count_parameters_in_millions(model):.2f}M')

    model.eval()
    image = torch.randn(1, 3, 480, 640)

    from torch.autograd import Variable

    image = Variable(image)
    with torch.no_grad():
        output = model(image)
        
    print(f'Input tensor shape: {image.shape}')
    print(f'Output (semantic) tensor shape: {output[0].shape}')
    print(f'Output (anomaly) tensor shape: {output[1].shape}')
