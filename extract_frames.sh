cd data
python build_rawframes.py ../../my_data/${DATASET}/videos_train/ ../../my_data/${DATASET}/rawframes_train/ --level 2 --flow-type tvl1 --ext mp4 --task both  --new-short 256
echo "Raw frames (RGB and tv-l1) Generated for train set"
