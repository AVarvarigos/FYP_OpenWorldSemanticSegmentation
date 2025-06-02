########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_v2.resnet import ResNet18, ResNet34, ResNet50
from src.models_v2.context_modules import get_context_module
from src.models_v2.model_utils import ConvBNAct, Swish, Hswish
from src.models_v2.decoder import Decoder
from src.models_v2.tru_for_decoder import TruForDecoderHead


class OWSNetwork(nn.Module):
    def __init__(
        self,
        height=480,
        width=640,
        num_classes=37,
        encoder="resnet18",
        encoder_block="BasicBlock",
        channels_decoder=None,  # default: [128, 128, 128]
        pretrained_on_imagenet=True,
        pretrained_dir=None,
        activation="relu",
        input_channels=3,
        encoder_decoder_fusion="add",
        context_module="ppm",
        nr_decoder_blocks=None,  # default: [1, 1, 1]
        weighting_in_encoder="None",
        upsampling="bilinear",
        tru_for_decoder=False
    ):
        super(OWSNetwork, self).__init__()
        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.weighting_in_encoder = weighting_in_encoder
        self.tru_for_decoder = tru_for_decoder

        if activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ["swish", "silu"]:
            self.activation = Swish()
        elif activation.lower() == "hswish":
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                "Only relu, swish and hswish as "
                "activation function are supported so "
                "far. Got {}".format(activation)
            )

        # encoder
        if encoder == "resnet18":
            self.encoder = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=False,  # pretrained_on_imagenet,
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
            raise NotImplementedError(
                "Only ResNets as encoder are supported "
                "so far. Got {}".format(activation)
            )

        # Freeze the backbone
        # TODO: Is this necessary?
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        self.channels_decoder_in = self.encoder.down_32_channels_out

        self.se_layer0 = nn.Identity()
        self.se_layer1 = nn.Identity()
        self.se_layer2 = nn.Identity()
        self.se_layer3 = nn.Identity()
        self.se_layer4 = nn.Identity()

        if encoder_decoder_fusion == "add":
            layers_skip1 = list()
            if self.encoder.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(
                    ConvBNAct(
                        self.encoder.down_4_channels_out,
                        channels_decoder[2],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(
                    ConvBNAct(
                        self.encoder.down_8_channels_out,
                        channels_decoder[1],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(
                    ConvBNAct(
                        self.encoder.down_16_channels_out,
                        channels_decoder[0],
                        kernel_size=1,
                        activation=self.activation,
                    )
                )
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if "learned-3x3" in upsampling:
            print(
                "Notice: for the context module the learned upsampling is "
                "not possible as the feature maps are not upscaled "
                "by the factor 2. We will use nearest neighbor "
                "instead."
            )
            upsampling_context_module = "nearest"
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = get_context_module(
            context_module,
            self.channels_decoder_in,
            channels_decoder[0],
            input_size=(height // 32, width // 32),
            activation=self.activation,
            upsampling_mode=upsampling_context_module,
        )

        # decoders
        self.decoder_ss = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes,
        )

        self.decoder_ow = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=19,
        )

        if self.tru_for_decoder:
            self.decoder_trufor = TruForDecoderHead(
                in_channels=[64, 128, 256, 512],
                dropout_ratio=0.1,
                norm_layer=nn.BatchNorm2d,
                embed_dim=768,
                align_corners=False,
                num_classes=num_classes
            )

    def forward(self, image):
        out = self.encoder.forward_first_conv(image)
        out = self.se_layer0(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        # block 1
        out_64 = self.encoder.forward_layer1(out)
        out_64 = self.se_layer1(out_64)
        skip1 = self.skip_layer1(out_64)

        # block 2
        out_128 = self.encoder.forward_layer2(out_64)
        out_128 = self.se_layer2(out_128)
        skip2 = self.skip_layer2(out_128)

        # block 3
        out_256 = self.encoder.forward_layer3(out_128)
        out_256 = self.se_layer3(out_256)
        skip3 = self.skip_layer3(out_256)

        # block 4
        out_512 = self.encoder.forward_layer4(out_256)
        out_512 = self.se_layer4(out_512)

        out = self.context_module(out_512)

        outs = [out, skip3, skip2, skip1]

        tru_for_result = None
        ow_output = self.decoder_ow(enc_outs=outs)
        if self.tru_for_decoder:
            input_tru_for = [out_64, out_128, out_256, out_512]
            tru_for_result = self.decoder_trufor(input_tru_for)
            ow_output = F.normalize(tru_for_result) + F.normalize(ow_output)

        return self.decoder_ss(enc_outs=outs), ow_output


def main():
    """
    Useful to check if model is built correctly.
    """
    model = OWSNetwork()
    print(model)

    model.eval()
    # rgb_image = torch.randn(1, 3, 480, 640)
    rgb_image = torch.randn(1, 3, 1080, 1920)

    from torch.autograd import Variable

    inputs_rgb = Variable(rgb_image)
    with torch.no_grad():
        output = model(inputs_rgb)
    print(output.shape)


if __name__ == "__main__":
    main()

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