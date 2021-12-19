_base_ = [
    './_base_/models/apn_threshold_i3d.py', './_base_/schedules/Adam_10e.py', './_base_/default_runtime.py'
]

# Change defaults
model = dict(backbone=dict(type='ResNet', pretrained='torchvision://resnet50', depth=50, norm_eval=False),
             cls_head=dict(num_classes=3, uncorrelated_progs='ignore'))

# input configuration
clip_len = 1
frame_interval = 1

# dataset settings
dataset_type = 'APNDataset'
data_root_train = 'my_data/DFMAD-70/Images/train'
data_root_val = 'my_data/DFMAD-70/Images/test'
ann_file_train = 'my_data/DFMAD-70/ann_train.csv'
ann_file_val = 'my_data/DFMAD-70/ann_test.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
# Even we won't use 'progression_label' and 'class_label' during val, we collect these two keys for distinguishing val from test
# Note that test is very different with val. In test we will apply trained model on untrimmed video for calculating mAP.
# While for val we apply trained model on trimmed video for calculating 'loss' or 'mae'.
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
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

# output settings
work_dir = './work_dirs/apn_coral_r50_1x1_10e_dfmad_rgb/'
output_config = dict(out=f'{work_dir}/results.json')
