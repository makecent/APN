_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN',
    backbone=dict(type='MViT2', pretrained=False, num_frames=16),
    cls_head=dict(
        type='APNHead',
        num_classes=20,
        in_channels=768,
        dropout_ratio=0.5,
        avg3d=False),
    train_cfg=dict(
        blending=dict(type='BatchAugBlendingProg', blendings=(dict(type='MixupBlendingProg', num_classes=20, alpha=.8),
                                                          dict(type='CutmixBlendingProg', num_classes=20, alpha=1.)))),
)

# input configuration
clip_len = 16
frame_interval = 4

# dataset settings
dataset_type = 'THUMOS14_IN'
data_root = 'my_data/thumos14'

data_train = (data_root + '/rawframes/train',
              data_root + '/rawframes/val')
data_val = data_root + '/rawframes/test'

ann_file_train = (data_root + '/annotations/apn/apn_train.csv',
                  data_root + '/annotations/apn/apn_val.csv')
ann_file_val = data_root + '/annotations/apn/apn_test.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleActionFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
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
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'raw_progression'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'raw_progression']),
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'raw_progression'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'raw_progression']),
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_train,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        clip_len=clip_len,
        frame_interval=frame_interval
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        untrimmed=True
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        untrimmed=True
    ))

# validation config
evaluation = dict(interval=250, metrics=['top_k_accuracy', 'MAE', 'mAP'], save_best='mAP', rule='greater')

# optimizer
optimizer = dict(type='AdamW',
                 lr=1e-4,   # 1.6e-3 is for batch-size=512
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
                 by_epoch=False,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=125,
                 warmup_by_epoch=True)
total_epochs = 1250

# runtime
checkpoint_config = dict(interval=250)
log_config = dict(interval=500)
fp16 = dict()
load_from = "work_dirs/apn_mvit2_32x4_10e_kinetics400_rgb/epoch_10.pth"
