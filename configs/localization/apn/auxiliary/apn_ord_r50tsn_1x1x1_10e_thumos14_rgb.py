num_classes = 20
clip_len = 1
num_clips = 1

model = dict(
    type='APN',
    pretrained='checkpoints/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='APNHead',
        num_classes=num_classes,
        spatial_type='avg2d',
        clip_len=clip_len))



# dataset settings
dataset_type = 'APNDataset'
data_root_train = ('my_data/thumos14/rawframes/train', 'my_data/thumos14/rawframes/val')
data_root_val = 'my_data/thumos14/rawframes/test'
ann_file_train = ('my_data/thumos14/ann_train.csv', 'my_data/thumos14/ann_val.csv')
ann_file_val = 'my_data/thumos14/ann_test.csv'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, num_clips=num_clips),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, num_clips=num_clips),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, num_clips=num_clips),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
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
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_root_train),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_root_val),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_root_val,
        untrimmed=True)
)

# optimizer
optimizer = dict(type='Adam', lr=1e-04)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)

# learning policy
# lr_config = dict(policy='fixed')
lr_config = dict(policy='step', step=10)
total_epochs = 10

# evaluation
evaluation = dict(interval=1, save_best='mae', metrics=['loss', 'mae'], dataset_name='Val')

# others
checkpoint_config = dict(interval=1)
log_config = dict(interval=500, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/apn_ord_r50tsn_1x1x1_10e_thumos14_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json')
