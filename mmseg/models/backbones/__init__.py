# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones

from .mix_transformer import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
                              mit_b3, mit_b4, mit_b5)
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d, ResNetClip
from .resnext import ResNeXt
from .vit import VisionTransformer
from .text_transformer import TextTransformer

__all__ = [
    'ResNet',
    'ResNetV1c',
    'ResNetV1d',
    'ResNetClip',
    'ResNeXt',
    'ResNeSt',
    'MixVisionTransformer',
    'mit_b0',
    'mit_b1',
    'mit_b2',
    'mit_b3',
    'mit_b4',
    'mit_b5',
    'VisionTransformer',
    'TextTransformer'
]
