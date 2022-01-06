_base_ = [
    './_base_/Adam_10e.py', './_base_/default_runtime.py',
]

# model settings
model = dict(
    type='APN',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='APNHead',
        num_classes=3,
        in_channels=2048,
        output_type='coral',
        loss=dict(type='ApnCORALLoss', uncorrelated_progs='random'),
        spatial_type='avg2d',
        dropout_ratio=0))

# input configuration
clip_len = 1
frame_interval = 1

# dataset settings
dataset_type = 'APNDataset'
data_root_train = 'my_data/dfmad70/rawframes/resized_train'
data_root_val = 'my_data/dfmad70/rawframes/resized_test'
ann_file_train = 'my_data/dfmad70/apn_train.csv'
ann_file_val = 'my_data/dfmad70/apn_test.csv'
ann_file_test = 'my_data/dfmad70/apn_test.csv'

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

# increase epoch to 15
total_epochs = 15

# output settings
work_dir = './work_dirs/apn_coral+random_r50_1x1_10e_dfmad_rgb/'
output_config = dict(out=f'{work_dir}/progressions.pkl')

# evaluation config
eval_config = dict(
    metric='mAP',
    metric_options=dict(
        mAP=dict(
            search=dict(
                min_e=80,
                max_s=20,
                min_L=600,
                method='mse'),
            nms=dict(iou_thr=0.4),
            dump_detections=f'{work_dir}/detections.pkl',
            dump_evaluation=f'{work_dir}/evaluation.json')))
