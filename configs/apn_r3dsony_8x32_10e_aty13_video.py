_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN',
    backbone=dict(
        type='ResNet3d_sony',
        init_cfg=dict(type='Pretrained',
                      checkpoint='https://github.com/hassony2/kinetics_i3d_pytorch/raw/master/model/model_rgb.pth'),
        modality='rgb'),
    cls_head=dict(
        type='APNHead',
        num_classes=200,
        in_channels=1024,
        dropout_ratio=0.5))

# input configuration
clip_len = 8
frame_interval = 32

# dataset settings
dataset_type = 'ActivityNet'
data_root = 'my_data/activitynet'

data_train = data_root + '/videos/train_resized'
data_val = data_root + '/videos/val_resized'

ann_file_train = data_root + '/annotations/apn/apn_aty_train_video.csv'
ann_file_val = data_root + '/annotations/apn/apn_aty_val_video.csv'

img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='DecordDecode'),
    dict(type='LabelToOrdinal'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
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
    videos_per_gpu=32,
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
        test_sampling=100,
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
evaluation = dict(interval=41750, metrics=['top_k_accuracy', 'MAE', 'mAP'], save_best='mAP', rule='greater', by_epoch=False)
checkpoint_config = dict(interval=41750, by_epoch=False)  # 41750 iter is half-epoch when batch_size=256

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=20))
# learning policy
lr_config = dict(policy='Fixed',
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=1,
                 warmup_by_epoch=True)
total_epochs = 10
fp16 = dict()
