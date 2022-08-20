from mmaction.datasets.pipelines import DecordInit, DecordDecode
results = dict(filename='/home/louis/PycharmProjects/APN/my_data/kinetics400/videos_val/stretching_arm/lVmBk_5ePYk_000061_000071.mp4')
t1 = DecordInit()
r = t1(results)
print('haha')