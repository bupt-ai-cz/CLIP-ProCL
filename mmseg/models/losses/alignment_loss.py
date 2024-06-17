# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Note that `downscale_label_ratio` method is adapted from: https://github.com/lhoyer/DAFormer

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor >= 1
    if scale_factor == 1:
        return gt.clone()
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # o/w next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out

def contrast_preparations(feat,
                          mask,
                          use_avg_pool,
                          scale_min_ratio,
                          num_classes,
                          ignore_index):
    # down-sample mask to fit feat
    if use_avg_pool:
        scale_factor = mask.shape[-1] // feat.shape[-1]
        mask = downscale_label_ratio(mask, scale_factor, scale_min_ratio, num_classes, ignore_index).long().detach()
    else:
        mask = F.interpolate(mask.float(), size=feat.shape[-2:], mode='nearest').long()
    # normalize the feat
    # feat = F.normalize(feat, p=2, dim=1)  # already normalized in proj_head.py
    # transpose the feat shape
    A = feat.size(1)
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, A)
    mask = mask.contiguous().view(-1)

    msk = (mask != ignore_index)
    # remove ignore_index pixels
    mask = mask[msk]
    feat = feat[msk]
    return feat, mask


def clip_contrastive(text_embedding,
                      feat,
                      mask,
                      weight=None,
                      contrast_temp=1.,
                      use_avg_pool=True,
                      scale_min_ratio=0.75,
                      num_classes=19,
                      reduction='none',
                      ignore_index=255):
    feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index)
    assert feat.requires_grad
    assert not mask.requires_grad

    proto_sim = feat.mm(text_embedding.permute(1, 0).contiguous()) / contrast_temp
    loss = F.cross_entropy(proto_sim, mask, weight=weight, reduction=reduction, ignore_index=ignore_index)

    return loss

@LOSSES.register_module()
class AlignmentLoss(nn.Module):
    """AlignmentLoss.

    Args:
        use_avg_pool (bool, optional): Whether to use average pooling for down sampling.
            Defaults to False.
        contrast_temp (double, optional): Temperature used in contrastive loss.
            Defaults to 1.
        reduction (str, optional): . Defaults to 'none'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 text_embedding_path,
                 use_avg_pool=False,
                 scale_min_ratio=0.75,
                 contrast_temp=1.,
                 reduction='none',
                 loss_weight=1.0,
                 num_classes=19):
        super(AlignmentLoss, self).__init__()
        self.use_avg_pool = use_avg_pool
        self.scale_min_ratio = scale_min_ratio
        self.contrast_temp = contrast_temp
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.contrast_criterion = clip_contrastive

        self.text_embedding = torch.load(text_embedding_path).cuda()


    def forward(self,
                feat,
                mask,
                weight=None,
                reduction_override=None,
                text_embedding_override=None,
                **kwargs):
        """Forward function."""
        # Parameters mean, covariance are sometimes required
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        text_embedding = text_embedding_override if text_embedding_override is not None else self.text_embedding

        loss_alignment = self.loss_weight * self.contrast_criterion(
            text_embedding,
            feat,
            mask,
            weight=weight,
            contrast_temp=self.contrast_temp,
            use_avg_pool=self.use_avg_pool,
            scale_min_ratio=self.scale_min_ratio,
            num_classes=self.num_classes,
            reduction=reduction)
        return loss_alignment