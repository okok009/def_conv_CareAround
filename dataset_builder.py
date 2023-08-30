import cv2
import numpy as np
from utils.detector_data_augmentation import Augmentation


# dataset路徑，也是aug後圖片存放的路徑，需要放絕對路徑
new_data_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/train350'

# 若設定img_size則會得到固定大小的Aug圖片，mean, std要先另外計算，則可以做正規化。
img_size = None
mean, std = None, None

# Information 的路徑，要按照指定規定
info_path = 'train2017.txt'
new_info_path = 'train350.txt'

# 原dataset 的圖片數量
original_data_num = 350

aug = Augmentation(img_size, mean, std)

with open(info_path, encoding='utf-8') as f:
    train_lines = f.readlines()[:original_data_num]

with open(new_info_path, 'r') as f:
    data_num = len(f.readlines())

for i in range(original_data_num):
    info = train_lines[i].split()
    img = cv2.imread(info[0])
    targets = np.array([list(map(np.int64,map(float, target.split(',')))) for target in info[1:]])
    bboxes = targets[:, 0:4]
    labels = targets[:, 4]

    while(True):
        img_a, bboxes_a, labels_a = aug(img, bboxes, labels)
        if len(bboxes_a) != 0:
            break

    info_a = []
    for j in range(len(bboxes_a)):
        info_a.append(f'{bboxes_a[j][0]},{bboxes_a[j][1]},{bboxes_a[j][2]},{bboxes_a[j][3]},{labels_a[j]}')

    with open(new_info_path, 'a') as f:
        f.write('\n'+ new_data_path + f'/{info[0][-16:]}' + ' ' + ' '.join(info_a[:]))
        f.write('\n'+ new_data_path + f'/aug_{i}.jpg' + ' ' + ' '.join(info_a[:]))

    cv2.imwrite(new_data_path + f'/{info[0][-16:]}', img)
    cv2.imwrite(new_data_path + f'/aug_{i}.jpg', img_a)
