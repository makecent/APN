import numpy as np
from matplotlib import pyplot as plt
from mmcv import load

from custom_modules.apn_utils import plot_gt, plot_prediction, plot_detection


# %% Load result files


def single_run():
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

# %% Compare two results files


def compare_run():
    progression1 = load("assert/mvit1/progression.pkl")
    detection1 = load("assert/mvit1/detection.pkl")

    progression2 = load("assert/mvit2_K400_pretrained/progression.pkl")
    detection2 = load("assert/mvit2_K400_pretrained/detection.pkl")

    # video_name = "video_test_0000006"
    all_videos = list(progression1.keys())
    for video_name in all_videos:
        f, (ax1, ax2) = plt.subplots(2)
        
        ax1.hlines(60, -50, 1050, ['r'])
        ax1.hlines(40, -50, 1050, ['r'])
        plot_gt(video_name, fig=ax1)
        plot_prediction(progression1, video_name, fig=ax1)
        plot_detection(detection1, video_name, fig=ax1)

        ax1.set_title("MViT1 "+video_name)
        ax1.grid()
        
        ax2.hlines(60, -50, 1050, ['r'])
        ax2.hlines(40, -50, 1050, ['r'])
        plot_gt(video_name, fig=ax2)
        plot_prediction(progression2, video_name, fig=ax2)
        plot_detection(detection2, video_name, fig=ax2)

        ax2.set_title("MViT2 "+video_name)
        ax2.grid()

        plt.setp([ax1, ax2],
                 yticks=np.arange(0, 101, 20.0),
                 xlabel='Frame Index',
                 ylabel='Progression')
        plt.tight_layout()
        plt.show()

compare_run()