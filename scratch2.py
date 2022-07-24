from custom_modules.apn_utils import plot_gt, plot_prediction, plot_detection
from matplotlib import pyplot as plt
from mmcv import load
import numpy as np

#%% Load result files
progression = load("/home/louis/PycharmProjects/APN/work_dirs/apn_mvit_16x8_10e_thumos14_rgb/progression.pkl")
detection = load("/home/louis/PycharmProjects/APN/work_dirs/apn_mvit_16x8_10e_thumos14_rgb/detection.pkl")

video_name = "video_test_0000006"
all_videos = list(progression.keys())
for video_name in all_videos:
    plt.figure(figsize=(15, 5))
    plt.hlines(60, -50, 1050, ['r'])
    plt.hlines(40, -50, 1050, ['r'])
    plot_gt(video_name)
    plot_prediction(progression, video_name)
    plot_detection(detection, video_name)

    plt.yticks(np.arange(0, 100, 20.0))
    plt.title(video_name)
    plt.xlabel('Frame Index')
    plt.ylabel('Progression')
    plt.grid()
    plt.show()
