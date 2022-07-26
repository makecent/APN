_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='MViT2'),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=768),
    train_cfg=dict(
        blending=dict(type='BatchAugBlending', blendings=(dict(type='MixupBlending', num_classes=400, alpha=.8),
                                                          dict(type='CutmixBlending', num_classes=400, alpha=1.))))
)

# input configuration
clip_len = 32
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
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
    dict(type='RandomErasing')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=10, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
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
evaluation = dict(metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW',
                 lr=1e-5,
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
fp16 = dict()
load_from = "work_dirs/apn_mvit2_32x4_10e_kinetics400_rgb/epoch_10.pth"
