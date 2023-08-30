import cv2
import numpy as np
import torch
import copy
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils.detector_data_augmentation import Augmentation, Original_Resize


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, batch_size, shuffle, device, input_shape = 600, train = True):
        self.annotation_lines   = annotation_lines
        self.batch              = batch_size
        self.shuffle            = shuffle
        self.device             = device
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.train              = train

        # self.reset()                                          因為在training_loop 有reset 了
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.length)
        else:
            self.index = np.arange(self.length)

    def __call__(self):
        resize = Original_Resize(self.input_shape, None, None)

        imgs_b = []
        bboxes_b = []
        labels_b = []

        to_tensor = transforms.ToTensor()
        
        iter = self.iteration
        batch_size = self.batch

        for i in self.index[iter * batch_size:(iter + 1) * batch_size]:
            info = self.annotation_lines[i].split()

            img = cv2.imread(info[0])
            img = img
            targets = np.array([list(map(np.int64,map(float, target.split(',')))) for target in info[1:]])
            bboxes = targets[:, 0:4]
            labels = targets[:, 4]

            img, bboxes, labels = resize(img, bboxes, labels)
            img = img / 255

            img = to_tensor(img)
            bboxes = torch.tensor(np.array(bboxes))
            labels = torch.tensor(labels)
            bboxes = bboxes.to(self.device)                         #bboxes跟labels要提前.to(device)
            labels = labels.to(self.device)                         #因為每張圖的bboxes數量不同，沒辦法合成tensor(也就無法合成後.to(device)，所以改成各自丟入gpu，再用list打包

            imgs_b.append(img)
            bboxes_b.append(bboxes)
            labels_b.append(labels)

        imgs_b = torch.stack([img for img in imgs_b], dim=0)
        imgs_b = imgs_b.to(self.device)                                       #img 因為大小都Resize過，可以包成tensor再全部丟入gpu。
        
        self.iteration += 1

        return imgs_b, bboxes_b, labels_b