# model settings
action_index = 0
num_stages = 100
clip_len = 1  # since we here use resnet50_2D as backbone, we train it on singe frame as input
model = dict(
    type='APN',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='APNHead',
        num_classes=1,
        fc_layers=(64, 32),
        num_stages=num_stages,
        in_channels=2048,
        spatial_type='avg2d',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'APNDataset'
data_root = 'my_data/DFMAD-70/Images/train'
data_root_val = 'my_data/DFMAD-70/Images/test'
ann_file_train = 'my_data/DFMAD-70/ann_train.csv'
ann_file_val = 'my_data/DFMAD-70/ann_test.csv'
ann_file_test = 'my_data/DFMAD-70/ann_test.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='FetchStackedFrames',
         clip_len=clip_len,
         frame_interval=1),
    dict(type='LabelToOrdinal', num_stages=num_stages),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFrames',
         clip_len=clip_len,
         frame_interval=1),
    dict(type='LabelToOrdinal', num_stages=num_stages),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
test_pipeline = [
    dict(type='FetchStackedFrames',
         clip_len=clip_len,
         frame_interval=1),
    dict(type='LabelToOrdinal', num_stages=num_stages),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        action_index=action_index,
        pipeline=train_pipeline,
        data_prefix=data_root),
    val=[dict(
        type=dataset_type,
        ann_file=ann_file_train,
        action_index=action_index,
        pipeline=val_pipeline,
        data_prefix=data_root),
        dict(
        type=dataset_type,
        ann_file=ann_file_val,
        action_index=action_index,
        pipeline=val_pipeline,
        data_prefix=data_root_val)],
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val))

# optimizer
optimizer = dict(type='Adam', lr=0.001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='fixed')

total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(interval=500, hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')])

evaluation = [dict(save_best=False, metrics=['loss', 'mae'], num_stages=num_stages, dataset_name='Train'),
              dict(save_best='mae', metrics=['loss', 'mae'], num_stages=num_stages, dataset_name='Val')]
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/apn_ac_r50_1x1x1_20e_dfmad_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json')
