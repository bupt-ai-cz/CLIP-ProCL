# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from xmlrpc.client import Boolean

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F

from mmseg.utils import get_root_logger
from ..builder import BACKBONES


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_mask=None):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)
        
        self.attn_mask = attn_mask

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, return_qkv=False):
        q, k, v = None, None, None
        if return_qkv:
            y = self.norm1(x)
            y = F.linear(y, self.attn.attn.in_proj_weight, self.attn.attn.in_proj_bias)
            N, L, C = y.shape
            y = y.view(N, L, 3, C//3).permute(2, 0, 1, 3).reshape(3*N, L, C//3)
            y = F.linear(y, self.attn.attn.out_proj.weight, self.attn.attn.out_proj.bias)
            q, k, v = y.tensor_split(3, dim=0)
            v += x
            v = self.ffn(self.norm2(v), identity=v)

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = self.attn(self.norm1(x), identity=x, attn_mask=self.attn_mask)
        x = self.ffn(self.norm2(x), identity=x)
        return x, q, k, v


@BACKBONES.register_module()
class TextTransformer(BaseModule):
    def __init__(self,
                 vocab_size=49408,
                 context_length=77,
                 embed_dims=512,
                 output_dims=512,
                 num_layers=12,
                 num_heads=8,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 pre_norm=False,
                 final_norm=True,
                 return_qkv=False,
                 skip_last_attn=False,
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(TextTransformer, self).__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.token_embedding = nn.Embedding(vocab_size, embed_dims)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, embed_dims))
        self.text_projection = nn.Parameter(torch.empty(embed_dims, output_dims))

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        attn_mask = torch.empty(context_length, context_length)
        attn_mask.fill_(float("-inf"))
        attn_mask.triu_(1)  # zero out the lower diagonal

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=False,
                    attn_mask=attn_mask))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=0)
            self.add_module(self.norm1_name, norm1)

        self.return_qkv = [False] * num_layers
        if isinstance(return_qkv, bool):
            for out_i in self.out_indices:
                self.return_qkv[out_i] = return_qkv
        elif isinstance(return_qkv, list) or isinstance(return_qkv, tuple):
            for i, out_i in enumerate(self.out_indices):
                self.return_qkv[out_i] = return_qkv[i]
        else:
            raise TypeError('return_qkv must be type of bool, list or tuple')

        self.skip_last_attn = skip_last_attn

    @property
    def ln_final(self):
        return getattr(self, self.norm1_name)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            print(self.load_state_dict(state_dict, False))
        elif self.init_cfg is not None:
            super(TextTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.positional_embedding, std=.02)
            trunc_normal_(self.text_projection, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, layer in enumerate(self.layers):
            x, q, k, v = layer(x, self.return_qkv[i] \
                                or (i==len(self.layers)-1 and self.skip_last_attn))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_with_pl(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, layer in enumerate(self.layers):
            x, q, k, v = layer(x, self.return_qkv[i] \
                                or (i==len(self.layers)-1 and self.skip_last_attn))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, inputs, pl_flag=False, extra_input=None):
        if pl_flag:
            return self.encode_text_with_pl(inputs, extra_input)
        else:
            return self.encode_text(inputs)

    def train(self, mode=True):
        super(TextTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()