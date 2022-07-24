_base_ = [
    './_base_/default_runtime.py'
]

# model settings
model = dict(
    type='APN_fly',
    flow_est=dict(
        type='PWCNet',
        encoder=dict(
            type='PWCNetEncoder',
            in_channels=3,
            net_type='Basic',
            pyramid_levels=[
                'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
            ],
            out_channels=(16, 32, 64, 96, 128, 196),
            strides=(2, 2, 2, 2, 2, 2),
            dilations=(1, 1, 1, 1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        decoder=dict(
            type='PWCNetDecoder',
            in_channels=dict(
                level6=81, level5=213, level4=181, level3=149, level2=117),
            flow_div=20.,
            corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            scaled=False,
            post_processor=dict(type='ContextNet', in_channels=565),
            flow_loss=dict(
                type='MultiLevelEPE',
                p=1,
                q=0.4,
                eps=0.01,
                reduction='sum',
                resize_flow='upsample',
                weights={
                    'level2': 0.005,
                    'level3': 0.01,
                    'level4': 0.02,
                    'level5': 0.08,
                    'level6': 0.32
                }),
        ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        init_cfg=dict(
            type='Kaiming',
            nonlinearity='leaky_relu',
            layer=['Conv2d', 'ConvTranspose2d'],
            mode='fan_in',
            bias=0)),
    backbone=dict(type='MViT', flow_input=True),
    cls_head=dict(
        type='APNHead',
        num_classes=20,
        in_channels=768,
        dropout_ratio=0.5,
        avg3d=False,
        pretrained="checkpoints/slowfast/MViTv2_B_32x3_k400_HEADONLY.pyth"),
    blending=dict(type='BatchAugBlendingProg', blendings=(dict(type='MixupBlendingProg', num_classes=400, alpha=.8),
                                                          dict(type='CutmixBlendingProg', num_classes=400, alpha=1.))),
)

# input configuration
clip_len = 17
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
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='LabelToOrdinal'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='pytorchvideo.RandAugment', magnitude=7, num_layers=4, prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'progression_label', 'class_label'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs', 'progression_label', 'class_label']),
    dict(type='RandomErasing')
]
val_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]
test_pipeline = [
    dict(type='FetchStackedFrames', clip_len=clip_len, frame_interval=frame_interval),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=()),
    dict(type='ToTensor', keys=['imgs']),
]

data = dict(
    videos_per_gpu=3,
    workers_per_gpu=4,
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
        modality='RGB',
        untrimmed=True
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

# validation config
evaluation = dict(metrics=['top_k_accuracy', 'MAE', 'mAP'], save_best='mAP', rule='greater')

# optimizer
optimizer = dict(type='AdamW',
                 lr=0.2e-3,  # 1.6e-3 is for batch-size=512
                 weight_decay=0.05,
                 paramwise_cfg=dict(
                     custom_keys={
                         '.backbone.model.cls_positional_encoding.cls_token': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_spatial': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_temporal': dict(decay_mult=0.0),
                         '.backbone.model.cls_positional_encoding.pos_embed_class': dict(decay_mult=0.0)}),
                 )
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr_ratio=0.01,
                 warmup='linear',
                 warmup_ratio=0.01,
                 warmup_iters=1,
                 warmup_by_epoch=True)
total_epochs = 10
# fp16 = dict()
