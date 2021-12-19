_base_ = [
    './_base_/models/apn_threshold_i3d.py', './_base_/schedules/Adam_10e.py', './_base_/default_runtime.py'
]

# Change defaults
model = dict(backbone=dict(pretrained='checkpoints/r3d_sony/model_flow.pth', modality='flow'),
             cls_head=dict(uncorrelated_progs='ignore'))

# input configuration
clip_len = 16
frame_interval = 1

# dataset settings
dataset_type = 'APNDataset'
data_root_train = ('my_data/thumos14/rawframes/train', 'my_data/thumos14/rawframes/val')
data_root_val = 'my_data/thumos14/rawframes/test'
ann_file_train = ('my_data/thumos14/ann_train.csv', 'my_data/thumos14/ann_val.csv')
ann_file_val = 'my_data/thumos14/ann_test.csv'

img_norm_cfg = dict(
    mean=[128, 128], std=[128, 128], to_bgr=False)

train_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_root_train,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow'
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow'
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
        untrimmed=True
    ))

# output settings
work_dir = './work_dirs/apn_coral_r3dsony_16x1_10e_thumos14_flow/'
output_config = dict(out=f'{work_dir}/results.json')
