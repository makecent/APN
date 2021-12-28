_base_ = [
    './_base_/apn_coral+random_i3d_rgb.py', './_base_/Adam_10e.py', './_base_/default_runtime.py'
]

# Change defaults
model = dict(cls_head=dict(num_classes=1, loss=dict(uncorrelated_progs='ignore')))

# input configuration
clip_len = 32
frame_interval = 4

# dataset settings
dataset_type = 'APNDataset'
data_root_train = 'my_data/dfmad70/rawframes/resized_train'
data_root_val = 'my_data/dfmad70/rawframes/resized_test'
ann_file_train = 'my_data/dfmad70/ann_train_ac1.csv'
ann_file_val = 'my_data/dfmad70/ann_test_ac1.csv'
ann_file_test = 'my_data/dfmad70/ann_test.csv'

img_norm_cfg = dict(
    mean=[128, 128, 128], std=[128, 128, 128], to_bgr=False)

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
    videos_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_root_train,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB'
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB'
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_test,
        pipeline=test_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        untrimmed=True
    ))

# output settings
work_dir = './work_dirs/apn_ac1_coral_r3dsony_32x4_10e_dfmad_rgb/'
output_config = dict(out=f'{work_dir}/progressions.pkl')
