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
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

_tokenizer = _Tokenizer()

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
        
        # visual
        self.text_channels = self.backbone_clip_cfg['text_channels']
        self.clip_channels = self.backbone_clip_cfg['clip_channels']
        self.clip_weights_path = self.backbone_clip_cfg['clip_weights_path']
        self.vit = self.backbone_clip_cfg.get('vit', False)
        self.feature_fusion = self.backbone_clip_cfg.get('feature_fusion', False)

        self.clip = builder.build_backbone(self.backbone_clip_cfg['clip_visual_cfg'])

        if self.vit:
            self.proj = nn.Conv2d(self.clip_channels, self.text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.k_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.v_proj = nn.Conv2d(self.clip_channels, self.clip_channels, 1)
            self.c_proj = nn.Conv2d(self.clip_channels, self.text_channels, 1)

        if self.feature_fusion:
            self.filter = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)

        # text
        self.text = builder.build_backbone(self.backbone_clip_cfg['clip_text_cfg'])
        self.n_ctx = self.backbone_clip_cfg['n_ctx']
        self.ctx_dim = self.text.ln0.weight.shape[0]
        dataset_name = self.backbone_clip_cfg['dataset_name']
        if dataset_name == 'cityscapes':
            from mmseg.datasets.cityscapes import CityscapesDataset
            self.classnames = CityscapesDataset.CLASSES
        self.n_cls = len(self.classnames)
        # random initialization
        self.ctx = nn.Parameter(self.init_ctx())  # to be optimized

        self.clip.cuda()
        self.clip.eval()
        self.text.cuda()
        self.text.eval()

    def init_ctx(self):
        ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim).cuda()
        nn.init.normal_(ctx_vectors, mean=0, std=0.02)

        self.ctx_init = "a photo of a"
        prompt = _tokenizer.encode(self.ctx_init)
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        token_embedding = loaded['text']['token_embedding.weight']
        embedding = token_embedding[prompt]

        ctx_vectors = resize(
            input=embedding.unsqueeze(0).unsqueeze(0),
            size=ctx_vectors.shape,
            mode='bilinear',
            align_corners=self.align_corners).squeeze(0).squeeze(0)

        self.prompt_prefix = " ".join(["X"] * self.n_ctx)
        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        return ctx_vectors

    def init_weights(self, call_super=True):
        if call_super:
            super(CLIPEncoderDecoderProjector, self).init_weights()
        self._load_clip_weights()
        self._init_text_embedding()

        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.text.parameters():
            param.requires_grad = False

        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            for param in current_attr.parameters():
                param.requires_grad = False

    def _load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.text.load_state_dict(loaded['text'])
        self.clip.load_state_dict(loaded['visual'])
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _init_text_embedding(self):
        classnames = [name.replace("_", " ") for name in self.classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = self.text.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

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
        # get text embedding
        self.text_embedding = self.prompt_learner()

        # get image feature
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
    
    def prompt_learner(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        self.text_embedding = self.text(prompts, pl_flag=True, extra_input=self.tokenized_prompts)
        self.text_embedding = self.text_embedding / self.text_embedding.norm(dim=-1, keepdim=True)

        return self.text_embedding


    def filter_feature(self, img, x):
        if self.filter is not None:
            x_ = []
            with torch.no_grad():
                clip_img_feats = self.clip(img)
                clip_output =  self.c_proj(self.v_proj(clip_img_feats[-1])) 
            clip_output = clip_output / clip_output.norm(dim=1, keepdim=True)

            if hasattr(self, 'text_embedding'):
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