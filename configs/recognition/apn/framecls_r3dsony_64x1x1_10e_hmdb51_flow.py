num_classes = 51
clip_len = 64
frame_interval = 1

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d_sony',
        pretrained='checkpoints/r3d_sony/model_flow.pth',
        modality='flow'),
    cls_head=dict(
        type='I3DHead',
        num_classes=num_classes,
        in_channels=1024,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = dict(untrimmed=False)
test_cfg = dict(untrimmed=True, average_clips='prob')

# dataset settings
dataset_type = 'RawframeDataset'
data_root_train = 'my_data/hmdb51/rawframes'
data_root_val = 'my_data/hmdb51/rawframes'
ann_file_train = 'my_data/hmdb51/annotations/split1/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'my_data/hmdb51/annotations/split1/hmdb51_val_split_1_rawframes.txt'

img_norm_cfg = dict(
    mean=[128, 128], std=[128, 128], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'label']),
]
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, num_clips=num_clips),
    dict(type='Rename', mapping={'class_label': 'label'}),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
        data_prefixes=data_root_train,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
    ),
    test=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=test_pipeline,
        data_prefixes=data_root_val,
        filename_tmpl='flow_{}_{:05}.jpg',
        modality='Flow',
        untrimmed=True,
    ))

# optimizer
optimizer = dict(type='Adam', lr=1e-04)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='fixed')
# lr_config = dict(policy='CosineRestart', restart_weights=[1, 1, 1, 1], periods=[2e04, 5e04, 5e04, 5e04], min_lr=5e-06, by_epoch=False)
total_epochs = 10

# evaluation
evaluation = dict(interval=1, save_best='top1_acc', metrics=['loss', 'framecls_top_k_accuracy', 'framecls_mean_class_accuracy'], dataset_name='Val')

# others
checkpoint_config = dict(interval=1)
log_config = dict(interval=500, hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/framecls_r3dsony_64x1x1_10e_hmdb51_flow/'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json')
