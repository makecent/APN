num_classes = 20
temporal_range = 64
frame_interval = 1
clip_len = int(temporal_range/frame_interval)

model = dict(
    type='APNonFeatures',
    num_classes=num_classes,
    feat_dim=512,
    output_type='coral',
    loss=dict(type='ApnCORALLoss', uncorrelated_loss='ignore'),
    dropout_ratio=0.5,)



# dataset settings
dataset_type = 'APNFeaturesDataset'
feat_train = 'my_data/thumos14/thumos14_tsp_features/validation/'
feat_val = 'my_data/thumos14/thumos14_tsp_features/test/'
ann_file_train = 'my_data/thumos14/ann_val.csv'
ann_file_val = 'my_data/thumos14/ann_test.csv'


train_pipeline = [
    dict(type='FetchStackedFeatures', feat_path=feat_train, clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='Collect', keys=['snippet', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['snippet', 'progression_label', 'class_label']),
]
val_pipeline = [
    dict(type='FetchStackedFeatures', feat_path=feat_val, clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='Collect', keys=['snippet', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['snippet', 'progression_label', 'class_label']),
]
test_pipeline = [
    dict(type='FetchStackedFeatures', feat_path=feat_val, clip_len=clip_len, frame_interval=frame_interval),
    dict(type='LabelToOrdinal'),
    dict(type='Collect', keys=['snippet'], meta_keys=()),
    dict(type='ToTensor', keys=['snippet']),
]


data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_files=ann_file_train,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_files=ann_file_val,
        pipeline=val_pipeline,
    ),
    # test=dict(
    #     type=dataset_type,
    #     ann_files=ann_file_val,
    #     pipeline=test_pipeline,
    #     data_prefixes=data_root_val,
    #     filename_tmpl='flow_{}_{:05}.jpg',
    #     modality='Flow',
    #     untrimmed=True
    # )
)

# optimizer
optimizer = dict(type='Adam', lr=1e-04)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='fixed')
# lr_config = dict(policy='CosineRestart', restart_weights=[1, 1, 1, 1], periods=[2e04, 5e04, 5e04, 5e04], min_lr=5e-06, by_epoch=False)
total_epochs = 10

# evaluation
evaluation = dict(interval=1, save_best='mae', metrics=['loss', 'mae'], dataset_name='Val')

# others
checkpoint_config = dict(interval=10)
log_config = dict(interval=500, hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/apn_coral_base_64x1_10e_th14tsp/'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json')
