# APN
This is the official repoistory of the paper _"Action Progression Network for Temporal Action Detection in Videos"_.
+ Up to **58\% mAP@0.5** on THUMOS14. To the best of our knowledge, currently,this is the highest mAP achieved on THUMOS14 dataset. 
+ Our method is **end-to-end**, no need of isolated operation of feature extraction, proposal generation, action classification ...
+ Easy to run. Our code is based on [mmaction2](https://github.com/open-mmlab/mmaction2).

## Build Environment
```shell
# 2080ti
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
pip install openmim
mim install mmaction2
conda install pandas h5py scipy
pip install future tensorboard

# RTX 3090
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
conda install pandas h5py
pip install future tensorboard
```

## Prepare Data
### THUMOS14
[Register](https://docs.google.com/forms/d/e/1FAIpQLScs9davISAtYQS7SEF5qQNu0jUpLzNH3aHmPfuqk2q1VYDkmw/viewform) on THUMOS14 to get the passwd. 
Download the processed dataset: [THUMOS14 download](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19074484r_connect_polyu_hk/Etnq-pgKYRhFj2G2WEsc33IBseulKby2Dhbhnc9K0BHZ1Q?e=4KqS4m).
+ Some wrong annotations are removed.
+ Only videos with temporal annotations (20 classes) are keeped.
+ Optical flows (TVL1) and RGB frames are included.

## Training

```shell
train.sh $CONFIG $GPUs $KWARGS
```
``$CONFIG``: Path of the configuration file (.py), which contains all the information (model setting, dataset setting, schedule..., all-in-one) required.\
``$GPUS``: Number of GPUs used for training. default: 2.
``$KWARGS``: Other arguments that used by [mmaction.tools.train.py](https://github.com/open-mmlab/mmaction2/blob/master/tools/train.py). Used: ``--validate``


<details>
<summary>Alternative</summary>
If you are familiar with mmlab repos and would like to run in a mmlab style, you may use the following cmd to start a training:

```shell
PYTHONPATH=$PWD:$PYTHONPATH mim train mmaction configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow.py --gpus 2 --validate --launcher pytorch
```
</details>

**Example**:
```shell
train.sh configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow.py 2
```
will train apn with threshold+randomness loss on thumos14 optical flows, with 32 frames input, temporal interval 4. Backbone I3D.
## Evaluation

```shell
bash test.sh $CONFIG $GPUS $CHECKPOINT $KWARGS
```
``$CONFIG``: Path of the configuration file (.py), which contains all the information (model setting, dataset setting, schedule..., all-in-one) required.\
``$GPUS``: Number of GPUs used for training. default: 2.
or with details
``CHECKPOINT``: Path of the saved trained models (``.pth``). default: work_dirs/CONFIG_STEM/latest.pth
``$KWARGS``: Other arguments that used by [mmaction.tools.train.py](https://github.com/open-mmlab/mmaction2/blob/master/tools/train.py). Used: ``--out "results.pkl" --eval mAP``.

<details>
<summary>Alternative</summary>
If you are familiar with mmlab repos and would like to run in a mmlab style, you may use the following cmd to start a training:

```shell
PYTHONPATH=$PWD:$PYTHONPATH mim test mmaction configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow.py --checkpoint ckpt.pth --gpus 2 --metrics mAP
```
</details>

**Example**:
```shell
test.sh configs/localization/apn/apn_coral+random_r3dsony_32x4_10e_thumos14_flow.py work_dirs/apn_coral+random_r3dsony_32x4_10e_thumos14_flow/latest.pth 2
```
will evalute the model of last epoch using two GPUs.
