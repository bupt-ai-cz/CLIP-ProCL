_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/deeplabv2red_r101-d8.py",
    "../_base_/datasets/uda_gta_to_cityscapes_512x512.py",
    # Basic UDA Self-Training
    "../_base_/uda/dacs.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm.py",
]
norm_cfg = dict(type='BN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Random Seed
seed = 42

# Dataset
crop_size = (640, 640)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 640)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Data RCS
data = dict(
    train=dict(
        source=dict(
            pipeline=gta_train_pipeline,
        ),
        target=dict(
            pipeline=cityscapes_train_pipeline
        ),
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)),
    val=dict(
        pipeline=test_pipeline
    ),
    test=dict(
        pipeline=test_pipeline
    )
)

# text + fusion
model = dict(
    type='CLIPEncoderDecoderProjector',
    auxiliary_head=dict(
        type='ProjHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=1,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='AlignmentLoss', use_avg_pool=True, num_classes=19, contrast_temp=1.0, loss_weight=1.0)),
    clip=dict(
        feature_fusion=True,
        text_channels=512,
        cls_bg=False,
        clip_channels=2048,
        clip_visual_cfg=dict(
            type='ResNetClip',
            depth=101,
            norm_cfg=norm_cfg,
            contract_dilation=True
        ),
        clip_text_cfg=dict(
            type='TextTransformer',
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
            norm_eval=False
        ),
        n_ctx=16,
        clip_weights_path='clip/RN101_clip_weights.pth',
        dataset_name='cityscapes')
    )

uda = dict(
    alpha=0.999,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    pl_flag=True,
    pl_weight=0.5,)

# Optimizer Hyper-parameters
optimizer_config = None
optimizer = dict(
    lr=6e-05, paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            ctx=dict(lr_mult=1.0)))
)
n_gpus = 1


# Schedule
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=-1)
evaluation = dict(interval=4000, metric='mIoU')

# Meta Information for Result Analysis
name = 'gta2cs_uda_warm_rcs_deeplabv2red_final'
exp = 'final'
name_dataset = 'gta2cityscapes'
name_architecture = 'Deeplabv2red'
name_encoder = 'ResNetV1c'
name_decoder = 'DLV2Head'
name_uda = 'dacs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
