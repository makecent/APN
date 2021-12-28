# model settings
num_stages = 100
clip_len = 10
action_index = 0  # indicate the action class you want to train
model = dict(
    type='APN',
    pretrained='checkpoints/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth',
    backbone=dict(
        type='ResNet3d',
        pretrained=None,
        pretrained2d=False,
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        non_local=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=False,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian'),
        zero_init_residual=False),
    cls_head=dict(
        type='APNHead',
        num_classes=1,
        num_stages=num_stages,
        fc_layers=(64, 32),
        in_channels=2048,
        spatial_type='avg3d',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'APNDataset'
data_root = 'my_data/dfmad70/rawframes/train'
data_root_val = 'my_data/dfmad70/rawframes/test'
ann_file_train = 'my_data/dfmad70/ann_train.csv'
ann_file_val = 'my_data/dfmad70/ann_test.csv'
ann_file_test = 'my_data/dfmad70/ann_test.csv'
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
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='FormatShape', input_format='NCTHW'),
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
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        action_index=action_index,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root),
    val=[dict(
        type=dataset_type,
        action_index=action_index,
        ann_file=ann_file_train,
        pipeline=val_pipeline,
        data_prefix=data_root),
        dict(
        type=dataset_type,
        action_index=action_index,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val)],
    test=dict(
        type=dataset_type,
        action_index=action_index,
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
work_dir = './work_dirs/apn_action0_coral_i3d_10x1x1_20e_dfmad_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/progressions.pkl')
