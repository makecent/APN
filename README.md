# APN
This is the official repoistory of the paper **[Progression-Guided Temporal Action Detection in Videos](https://arxiv.org/abs/2308.09268)**. Our model achieves **58\% mAP@0.5** on THUMOS14 in **end-to-end** manner.

![apn_framework_v2](https://user-images.githubusercontent.com/42603768/167289156-e1748bc1-a8e1-4bcf-93f8-2ae0e2dc4d99.png)


# Build Environment
```shell
# 2080ti
conda create -n open-mmlab -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
conda install pandas h5py scipy
pip install openmim future tensorboard future timm pytorchvideo
mim install mmengine mmaction2 mmdet 
```

# Prepare Data
Download the pre-processed THUMOS14 [raw frames](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19074484r_connect_polyu_hk/Eue7ALiNQ1NDteHBZXiLuv8B_Q1eM0OMQt1tY3-yHWylGw?e=5ZNyhG) and [the annotations (APN format)](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19074484r_connect_polyu_hk/EqklxouUfZpDgLSQRWEfT4MBcCg1VS6PaJKtjEu8kXAcMw?e=0Qo4ys), and put them under the repo root. You are suggested to put the data in other palce (SSD would be best) and set a symbolic link here pointing to the data path.
The folder structure should be like:
```shell
APN
|-- configs
|-- ...
|-- my_data
|   |-- thumos14
|   |   |-- annotations
|   |   |   |-- apn
|   |   |   |   |-- apn_train.csv
|   |   |   |   |-- apn_val.csv
|   |   |   |   |-- apn_test.csv
|   |   |-- rawframes
|   |   |   |-- train
|   |   |   |   |-- v_BaseballPitch_g01_c01
|   |   |   |   |   |-- img_00000.jpg
|   |   |   |   |   |-- img_00001.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- img_00106.jpg
|   |   |   |   |   |-- flow_x_00000.jpg
|   |   |   |   |   |-- flow_x_00001.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- flow_x_00105.jpg
|   |   |   |   |   |-- flow_y_00000.jpg
|   |   |   |   |   |-- flow_y_00001.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- flow_y_00105.jpg
|   |   |   |   |-- ...
|   |   |   |-- val
|   |   |   |   |-- video_validation_0000051
|   |   |   |   |-- ...
|   |   |   |-- test
|   |   |   |   |-- video_test_0000004
|   |   |   |   |-- ...
```
+ Optical flows (TVL1) and RGB frames are included.
+ Only videos with temporal annotations (20 classes) are keeped.
+ Some wrong annotated videos are removed.

# Training
**Let's take as example the implementation of APN on THUMOS14 of optical flow using I3D as backbone with resolution of (32 frames x 4 stride):**
```shell
train.sh configs/localization/apn/apn_r3dsony_32x4_10e_thumos14_flow 2
```
\*replace the `2` with the number of GPUs you want use.

# Test
After the training finished, you may use the below command to test the trained checkpoint.

```shell
test.sh configs/localization/apn/apn_r3dsony_32x4_10e_thumos14_flow.py work_dirs/apn_r3dsony_32x4_10e_thumos14_flow/latest.pth 2
```
\*replace the `2` with the number of GPUs you want use.

# Acknowledgement
Our code is based on the [MMAction2](https://github.com/open-mmlab/mmaction2).

# Citation
If you find our work useful, please cite:
```latex
@article{lu2023progression,
  title={Progression-Guided Temporal Action Detection in Videos},
  author={Lu, Chongkai and Mak, Man-Wai and Li, Ruimin and Chi, Zheru and Fu, Hong},
  journal={arXiv preprint arXiv:2308.09268},
  year={2023}
}
```
