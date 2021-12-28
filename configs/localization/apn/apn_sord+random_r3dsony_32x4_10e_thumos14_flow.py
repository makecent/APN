_base_ = [
    './_base_/apn_coral+random_i3d_flow.py', './_base_/Adam_10e.py',
    './_base_/default_runtime.py', './_base_/thumos14_flow.py'
]

# Change defaults
model = dict(cls_head=dict(output_type='classification', loss=dict(type='ApnSORDLoss')))

# output settings
work_dir = './work_dirs/apn_sord+random_r3dsony_32x4_10e_thumos14_flow/'
output_config = dict(out=f'{work_dir}/progressions.pkl')
