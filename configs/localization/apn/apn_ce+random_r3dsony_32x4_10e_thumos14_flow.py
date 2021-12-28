_base_ = [
    './_base_/apn_coral+random_i3d_flow.py', './_base_/Adam_10e.py',
    './_base_/default_runtime.py', './_base_/thumos14_flow.py'
]

# Change defaults
model = dict(cls_head=dict(output_type='classification', loss=dict(type='ApnCELoss')))
optimizer = dict(lr=1e-5)

# output settings
work_dir = './work_dirs/apn_ce+random_r3dsony_32x4_10e_thumos14_flow/'
output_config = dict(out=f'{work_dir}/progressions.pkl')

# evaluation
evaluation = dict(metrics=['MAE'],
                  save_best='MAE',
                  rule='less',
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