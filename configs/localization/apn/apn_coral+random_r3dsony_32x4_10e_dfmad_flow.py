_base_ = [
    './_base_/apn_coral+random_i3d_flow.py',
    './_base_/dfmad70_flow.py',
    './_base_/Adam_10e.py',
    './_base_/default_runtime.py'
]

# Change defaults
model = dict(cls_head=dict(num_classes=3))

# runtime settings
work_dir = './work_dirs/apn_coral+random_r3dsony_32x4_10e_dfmad_flow/'
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
