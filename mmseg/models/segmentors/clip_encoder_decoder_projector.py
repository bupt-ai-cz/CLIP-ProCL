# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from mmcv.utils import print_log
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.utils import get_root_logger
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

@SEGMENTORS.register_module()
class CLIPEncoderDecoderProjector(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 clip=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLIPEncoderDecoderProjector, self).__init__(backbone, decode_head, neck, auxiliary_head,
                                                 train_cfg, test_cfg, pretrained, init_cfg) 
        self.filter = None
        self._init_clip(clip)
        
    def _init_clip(self, clip):
        # https://github.com/chongzhou96/MaskCLIP/blob/master/mmseg/models/decode_heads/maskclip_plus_head.py
        self.backbone_clip_cfg = deepcopy(clip)
        self.text_channels = self.backbone_clip_cfg['text_channels']
        self.clip_channels = self.backbone_clip_cfg['clip_channels']
        self.clip_weights_path = self.backbone_clip_cfg['clip_weights_path']
        self.vit = self.backbone_clip_cfg.get('vit', False)
        self.feature_fusion = self.backbone_clip_cfg.get('feature_fusion', False)

        self.text_embedding = torch.load(self.backbone_clip_cfg['text_embedding_path']).cuda()
        self.text_embedding = nn.Parameter(self.text_embedding)   # to be optimized

        self.clip = builder.build_backbone(self.backbone_clip_cfg['clip_cfg'])

        if self.vit:
            self.proj = nn.Conv2d(self.clip_channels, self.text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.k_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.v_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.c_proj = nn.Conv2d(self.clip_channels, self.text_channels, 1)

        if self.feature_fusion:
            self.filter = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)

        self.clip.cuda()
        self.clip.eval()

    def init_weights(self, call_super=True):
        if call_super:
            super(CLIPEncoderDecoderProjector, self).init_weights()
        self._load_clip_weights()

        for param in self.clip.parameters():
            param.requires_grad = False

        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            for param in current_attr.parameters():
                param.requires_grad = False

    def _load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.clip.load_state_dict(loaded['clip'])
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None,
                                      **kwargs):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight, **kwargs)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg, **kwargs)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)

        # clip image feature fusion
        x, _ = self.filter_feature(img, x)

        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        # clip image feature fusion
        x, clip_output = self.filter_feature(img, x)
        clip_output = F.interpolate(clip_output, size=img.shape[-2:], mode='bilinear', align_corners=True)
        losses['clip_output'] = clip_output

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight, text_embedding_override=self.text_embedding.detach())
            losses.update(loss_aux)

        return losses
    
    def filter_feature(self, img, x):
        if self.filter is not None:
            x_ = []
            with torch.no_grad():
                clip_img_feats = self.clip(img)
                clip_output =  self.c_proj(self.v_proj(clip_img_feats[-1])) 
                clip_output = clip_output / clip_output.norm(dim=1, keepdim=True)

            clip_output = F.conv2d(clip_output, self.text_embedding[:, :, None, None])

            for i in range(len(clip_img_feats)):
                if i == len(clip_img_feats) - 1:
                    x_filter = self.filter(x[i] * F.normalize(F.interpolate(clip_img_feats[i], size=x[i].shape[-2:], mode='bilinear', align_corners=True), dim=1))
                    x_.append(x[i] * x_filter)
                else:
                    x_.append(x[i])
            x = tuple(x_)
        else:
            x = x

        return x, clip_output