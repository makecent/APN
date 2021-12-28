_base_ = [
    './_base_/apn_coral+random_i3d_rgb.py', './_base_/Adam_10e.py', './_base_/default_runtime.py', './_base_/thumos14_videos.py'
]

# Change defaults


# output settings
# work_dir = './work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_flow/'
work_dir = './work_dirs/Test/'
output_config = dict(out=f'{work_dir}/results.pkl')
