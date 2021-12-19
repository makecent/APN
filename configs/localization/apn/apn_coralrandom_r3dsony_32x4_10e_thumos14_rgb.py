_base_ = [
    './_base_/models/apn_threshold_i3d.py', './_base_/schedules/Adam_10e.py',
    './_base_/default_runtime.py', './_base_/thumos14_rawframes.py'
]

# Change defaults



# output settings
# work_dir = './work_dirs/apn_coralrandom_r3dsony_32x4_10e_thumos14_rgb/'
work_dir = './work_dirs/Test/'
output_config = dict(out=f'{work_dir}/results.pkl')
