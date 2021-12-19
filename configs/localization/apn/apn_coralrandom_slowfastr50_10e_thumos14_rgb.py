_base_ = [
    './_base_/schedules/Adam_10e.py', './_base_/default_runtime.py'
]

# Change defaults
# slow: 32 x 4; fast: 128 x 1
model = dict(
     type='APN',
     pretrained='checkpoints/slowfast/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb_20200810-863812c2.pth',
     backbone=dict(
         type='ResNet3dSlowFast',
         pretrained=None,
         resample_rate=4,  # tau
         speed_ratio=4,  # alpha
         channel_ratio=8,  # beta_inv
         slow_pathway=dict(
             type='resnet3d',
             depth=50,
             pretrained=None,
             lateral=True,
             conv1_kernel=(1, 7, 7),
             dilations=(1, 1, 1, 1),
             conv1_stride_t=1,
             pool1_stride_t=1,
             inflate=(0, 0, 1, 1),
             norm_eval=False),
         fast_pathway=dict(
             type='resnet3d',
             depth=50,
             pretrained=None,
             lateral=False,
             base_channels=8,
             conv1_kernel=(5, 7, 7),
             conv1_stride_t=1,
             pool1_stride_t=1,
             norm_eval=False)),
     cls_head=dict(
         type='APNHead',
         num_classes=20,
         in_channels=2304,
         output_type='coral',
         loss=dict(type='ApnCORALLoss', uncorrelated_progs='random'),
         dropout_ratio=0.5,
         spatial_type='avg3d'))

# input configuration
num_classes = 20
clip_len = 128  # T x tau x alpha,
frame_interval = 1  # alpha

# dataset settings
dataset_type = 'APNDataset'
data_root_train = ('my_data/thumos14/rawframes/train', 'my_data/thumos14/rawframes/val')
data_root_val = 'my_data/thumos14/rawframes/test'
ann_file_train = ('my_data/thumos14/ann_train.csv', 'my_data/thumos14/ann_val.csv')
ann_file_val = 'my_data/thumos14/ann_test.csv'

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
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_root_train,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        modality='RGB',
        untrimmed=True
    ))

# output settings
work_dir = './work_dirs/apn_coralrandom_slowfastr50_32x4_128x1_10e_thumos14_rgb/'
output_config = dict(out=f'{work_dir}/results.json')
