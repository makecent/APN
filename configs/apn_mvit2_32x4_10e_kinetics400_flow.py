_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN_fly',
    flow_est=dict(
        type='PWCNet',
        encoder=dict(
            type='PWCNetEncoder',
            in_channels=3,
            net_type='Basic',
            pyramid_levels=[
                'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
            ],
            out_channels=(16, 32, 64, 96, 128, 196),
            strides=(2, 2, 2, 2, 2, 2),
            dilations=(1, 1, 1, 1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        decoder=dict(
            type='PWCNetDecoder',
            in_channels=dict(
                level6=81, level5=213, level4=181, level3=149, level2=117),
            flow_div=20.,
            corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            scaled=False,
            post_processor=dict(type='ContextNet', in_channels=565),
            flow_loss=dict(
                type='MultiLevelEPE',
                p=1,
                q=0.4,
                eps=0.01,
                reduction='sum',
                resize_flow='upsample',
                weights={
                    'level2': 0.005,
                    'level3': 0.01,
                    'level4': 0.02,
                    'level5': 0.08,
                    'level6': 0.32
                }),
        ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        init_cfg=dict(
            type='Kaiming',
            nonlinearity='leaky_relu',
            layer=['Conv2d', 'ConvTranspose2d'],
            mode='fan_in',
            bias=0)),
    backbone=dict(type='MViT2', flow_input=True),
    cls_head=dict(
        type='APNHead',
        num_classes=400,
        in_channels=768,
        dropout_ratio=0.5,
        avg3d=False,
        pretrained="checkpoints/slowfast/MViTv2_B_32x3_k400_HEADONLY.pyth"),
    blending=dict(type='BatchAugBlendingProg', blendings=(dict(type='MixupBlendingProg', num_classes=400, alpha=.8),
                                                          dict(type='CutmixBlendingProg', num_classes=400, alpha=1.))),
)

# input configuration
clip_len = 33
frame_interval = 4

# dataset settings
dataset_type = 'VideoDataset_MAE'
data_root = 'my_data/kinetics400/videos_train'
data_root_val = 'my_data/kinetics400/videos_val'
ann_file_train = 'my_data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'my_data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'my_data/kinetics400/kinetics400_val_list_videos.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='LabelToOrdinal'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='pytorchvideo.RandAugment', magnitude=7, num_layers=4, prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Rename', mapping=dict(label='class_label')),
    dict(type='Collect', keys=['imgs', 'class_label', 'progression_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'class_label', 'progression_label']),
    dict(type='RandomErasing')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='LabelToOrdinal'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'raw_progression'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'raw_progression'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=10, test_mode=True),
    dict(type='LabelToOrdinal'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'raw_progression'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'raw_progression']),
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=6,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# validation config
evaluation = dict(metrics=['top_k_accuracy', 'mean_class_accuracy', 'MAE'], save_best='MAE', rule='less')

# optimizer
optimizer = dict(type='AdamW',
                 lr=1e-4,
                 weight_decay=0.05,
                 paramwise_cfg=dict(
                     custom_keys={
                         '.backbone.model.cls_positional_encoding.cls_token': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_spatial': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_temporal': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_class': dict(decay_mult=0.0)}),
                 )
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=0.01,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=1,
                 warmup_by_epoch=True)
total_epochs = 10
# fp16 = dict()
