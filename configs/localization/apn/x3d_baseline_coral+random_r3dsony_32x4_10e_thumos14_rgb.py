_base_ = [
    './_base_/apn_coral+random_i3d_rgb.py', './_base_/Adam_10e.py',
    './_base_/default_runtime.py', './_base_/thumos14_rgb.py'
]

# model settings
model = dict(
    type='APN',
    backbone=dict(
        _delete_=True,
        type='X3D',
        gamma_w=1,
        gamma_b=2.25,
        gamma_d=2.2),
    cls_head=dict(in_channels=432))
data = dict(videos_per_gpu=4)
# output settings
work_dir = './work_dirs/x3d_baseline_coral+random_r3dsony_32x4_10e_thumos14_rgb/'
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
