import numpy as np


def one_vs_n_iou(one_box, n_boxes):
    """Compute IoU score between a box and other n boxes.

    (1D)
    """
    one_len = one_box[1] - one_box[0]
    n_len = n_boxes[:, 1] - n_boxes[:, 0]
    inter_left = np.maximum(one_box[0], n_boxes[:, 0])
    inter_right = np.minimum(one_box[1], n_boxes[:, 1])
    inter_len = (inter_right - inter_left).clip(min=0)
    union_len = one_len + n_len - inter_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


apn_test = "/home/louis/PycharmProjects/APN/my_data/thumos14/annotations/apn/apn_test.csv"
with open(apn_test, 'r') as f:
    t = f.readlines()

rt = []
cur = "video_test_0000004.mp4"
cur_boxes = np.empty([0, 2])
removed = []
strange = []
for i in t:
    passed = True
    l = i.strip().split(',')
    if l[0] == cur:
        this_box = np.array([int(l[2]), int(l[3])])
        ious = one_vs_n_iou(this_box, cur_boxes)
        if (ious > 0).any():
            passed = False
            removed.append(i)
        if (ious > 0.5).any():
            if not (l[-1] == '4' or l[-1] == '7'):
                strange.append(i)
        if passed:
            cur_boxes = np.vstack([cur_boxes, this_box[None, ...]])
    else:
        cur = l[0]
        cur_boxes = np.empty([0, 2])
    if passed:
        rt.append(i)

print('The End')