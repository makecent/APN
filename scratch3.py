from mmaction.datasets.pipelines import DecordInit, DecordDecode

results = dict(filename='my_data/kinetics400/videos_val/abseiling/y3-3i88mNPc_000008_000018.mp4')
init = DecordInit()
decode = DecordDecode()
results = init(results)
results = decode(results)
print(results)
