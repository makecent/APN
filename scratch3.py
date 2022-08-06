import decord

video_reader = decord.VideoReader('/home/louis/PycharmProjects/APN/my_data/kinetics400/videos_val/barbequing/hJoIok6KhSw_000554_000564.mp4')

imgs = list(video_reader.get_batch([3, 3 + 1]).asnumpy())
print()