# imports
custom_imports = dict(imports=['my_models', 'dataloader'], allow_failed_imports=False)

# others
checkpoint_config = dict(interval=1)
log_config = dict(interval=500, hooks=[dict(type='TensorboardLoggerHook'), dict(type='TextLoggerHook')])

# evaluation
evaluation = dict(rule='less', save_best='mae', metrics=['loss', 'mae'])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


