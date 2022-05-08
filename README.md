# APN
This is the official repoistory of the paper **"Action Progression Network for Temporal Action Localization in Videos"**. Our model achives **58\% mAP@0.5** on THUMOS14 in **end-to-end** manner.

![apn_framework_v2](https://user-images.githubusercontent.com/42603768/167289156-e1748bc1-a8e1-4bcf-93f8-2ae0e2dc4d99.png)


# Build Environment
```shell
# 2080ti
conda create -n open-mmlab -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
pip install openmim
mim install mmaction2
conda install pandas h5py scipy
pip install future tensorboard
```

# Prepare Data
[Sign on](https://docs.google.com/forms/d/e/1FAIpQLScs9davISAtYQS7SEF5qQNu0jUpLzNH3aHmPfuqk2q1VYDkmw/viewform) the THUMOS14 to get the passwd. 
Download the our processed dataset: [THUMOS14 download](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19074484r_connect_polyu_hk/Etnq-pgKYRhFj2G2WEsc33IBseulKby2Dhbhnc9K0BHZ1Q?e=4KqS4m).

+ Optical flows (TVL1) and RGB frames are included.
+ Only videos with temporal annotations (20 classes) are keeped.
+ Some wrong annotations are removed.

# Training

```shell
train.sh configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow 2
```
\*replace the `2` with the number of GPUs you want use.

# Test
The above training already includes a test (after the training finished). In case of some error, you may use the below command to test the trained checkpoint.

```shell
test.sh configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow.py work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_flow/latest.pth 2
```
\*replace the `2` with the number of GPUs you want use.

# Acknowledgement
Our code is based on the [MMAction2](https://github.com/open-mmlab/mmaction2).
