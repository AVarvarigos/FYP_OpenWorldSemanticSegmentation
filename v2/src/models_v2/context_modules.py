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

from src.models_v2.neck import AdaptivePyramidPoolingModule, PyramidPoolingModule


def get_context_module(
    context_module_name,
    channels_in,
    channels_out,
    input_size,
    activation,
    upsampling_mode="bilinear",
):
    if "appm" in context_module_name:
        if context_module_name == "appm-1-2-4-8":
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = AdaptivePyramidPoolingModule(
            channels_in,
            channels_out,
            bins=bins,
            input_size=input_size,
            activation=activation,
            upsampling_mode=upsampling_mode,
        )
        channels_after_context_module = channels_out
    elif "ppm" in context_module_name:
        if context_module_name == "ppm-1-2-4-8":
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = PyramidPoolingModule(
            channels_in,
            channels_out,
            bins=bins,
            activation=activation,
            upsampling_mode=upsampling_mode,
        )
        channels_after_context_module = channels_out
    else:
        context_module = nn.Identity()
        channels_after_context_module = channels_in
    return context_module, channels_after_context_module
