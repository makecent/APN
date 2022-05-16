_base_ = [
    './_base_/Adam_10e.py',
    './_base_/default_runtime.py',
]
# model settings
model = dict(
    type='APN',
    backbone=dict(
        type='X3D',
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2),
    cls_head=dict(
        type='APNHead',
        num_classes=20,
        in_channels=432,
        output_type='coral',
        loss=dict(type='ApnCORALLoss', uncorrelated_progs='random'),
        spatial_type='avg3d',
        dropout_ratio=0.5))

# input configuration
clip_len = 16
frame_interval = 5

# dataset settings
dataset_type = 'APNDataset'
data_root = 'my_data/thumos14'

data_train = (data_root + '/rawframes/train',
              data_root + '/rawframes/val')
data_val = data_root + '/rawframes/test'

ann_file_train = (data_root + '/annotations/apn/apn_train.csv',
                  data_root + '/annotations/apn/apn_val.csv')
ann_file_val = data_root + '/annotations/apn/apn_test.csv'

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
    videos_per_gpu=10,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_train,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB'
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB'
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

# output settings
work_dir = './work_dirs/x3d_baseline_coral+random_r3dsony_16x5_10e_thumos14_rgb/'
output_config = dict(out=f'{work_dir}/progressions.pkl')

# evaluation config
eval_config = dict(
    metric_options=dict(
        mAP=dict(
            search=dict(
                min_e=60,
                max_s=40,
                min_L=60,
                method='mse'),
            nms=dict(iou_thr=0.4),
            dump_detections=f'{work_dir}/detections.pkl',
            dump_evaluation=f'{work_dir}/evaluation.json')))
load_from = "https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"
