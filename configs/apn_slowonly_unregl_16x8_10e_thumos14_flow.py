_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False,
        in_channels=2,
        with_pool2=False),
    cls_head=dict(
        type='APNHead',
        loss_cls=dict(type='CrossEntropyLossV2', label_smoothing=0.0),
        loss_reg=dict(type='BCELossWithLogitsV2', label_smoothing=0.0),
        num_classes=20,
        in_channels=2048,
        dropout_ratio=0.5),
)

# input configuration
clip_len = 16
frame_interval = 8

# dataset settings
dataset_type = 'THUMOS14'
data_root = 'my_data/thumos14'

data_train = (data_root + '/rawframes/train',
              data_root + '/rawframes/val')
data_val = data_root + '/rawframes/test'

ann_file_train = (data_root + '/annotations/apn/apn_train.csv',
                  data_root + '/annotations/apn/apn_val.csv')
ann_file_val = data_root + '/annotations/apn/apn_test.csv'

img_norm_cfg = dict(
    mean=[128, 128], std=[128, 128], to_bgr=False)

train_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='LabelToOrdinal'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_train,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
        untrimmed=True
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
        untrimmed=True
    ))

# validation config
evaluation = dict(metrics=['top_k_accuracy', 'MAE', 'mAP'], save_best='mAP', rule='greater')

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=20))
# learning policy
lr_config = dict(policy='Fixed',
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=1,
                 warmup_by_epoch=True)
total_epochs = 10
fp16 = dict()
load_from = "https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth"
