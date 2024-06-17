# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from mmseg.ops import resize
from ..losses import accuracy

@HEADS.register_module()
class ProjHead(BaseDecodeHead):
    """Projection Head for feature dimension reduction in contrastive loss.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        padding (int): The padding for convs in the head. Default: 1.
    """
    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 padding=1,
                 **kwargs):
        assert num_convs in (1, 2)
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        super(ProjHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        self.mid_channels = self.in_channels

        convs = []
        if num_convs > 1:
            convs.append(
                ConvModule(
                    self.in_channels,
                    self.mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        convs.append(
            ConvModule(
                self.mid_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=padding,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = F.normalize(self.convs(x), p=2, dim=1)
        return output

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight, **kwargs)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None, **kwargs):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)

        loss['loss_alignment'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index,
            **kwargs)
        return loss
