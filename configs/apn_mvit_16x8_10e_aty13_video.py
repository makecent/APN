_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN',
    backbone=dict(type='MViTB'),
    cls_head=dict(
        type='APNHead',
        num_classes=200,
        in_channels=768,
        dropout_ratio=0.5,
        avg3d=False),
    blending=dict(type='BatchAugBlendingProg', blendings=(dict(type='MixupBlendingProg', num_classes=200, alpha=.8),
                                                          dict(type='CutmixBlendingProg', num_classes=200, alpha=1.))),
)

# input configuration
clip_len = 16
frame_interval = 8

# dataset settings
dataset_type = 'ActivityNet'
data_root = 'my_data/activitynet'

data_train = data_root + '/videos/train_resized'
data_val = data_root + '/videos/val_resized'

ann_file_train = data_root + '/annotations/apn/apn_aty_train_video.csv'
ann_file_val = data_root + '/annotations/apn/apn_aty_val_video.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='pytorchvideo.RandAugment', magnitude=7, num_layers=4, prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
    dict(type='RandomErasing')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_train,
        filename_tmpl='img_{:05}.jpg',
        modality='Video'
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_val,
        filename_tmpl='img_{:05}.jpg',
        modality='Video',
        untrimmed=True
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_val,
        filename_tmpl='img_{:05}.jpg',
        modality='Video',
        untrimmed=True
    ))

# validation config
evaluation = dict(metrics=['top_k_accuracy', 'MAE', 'mAP'], save_best='mAP', rule='greater')

# optimizer
optimizer = dict(type='AdamW', lr=0.2e-3, weight_decay=0.05)  # this lr is for batch-size=64
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=0.01,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=1,
                 warmup_by_epoch=True)
total_epochs = 10
fp16 = dict()
