import torch
import cv2
from torchvision import transforms
from nets.fasterrcnn_resnet50_fpn import fasterrcnn_resnet50_fpn_def, fasterrcnn_resnet50_fpn_predict
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

import torch.nn as nn
import numpy as np


# weights = 'checkpoints/ep40-loss0.5875978469848633.pth'
# model = fasterrcnn_resnet50_fpn_predict(weights)
# model.eval()

# weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
# model = fasterrcnn_resnet50_fpn(weights)
# model.eval()

weights = 'checkpoints/d_ep40-loss0.8746699690818787.pth'
model = fasterrcnn_resnet50_fpn_def(weights)
model.eval()

img = 'img/street.jpg'
img = cv2.imread(img)

img[:, :, ::1] = img[:, :, ::-1]
img1 = np.copy(img)

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t = torch.unsqueeze(img_t, dim=0)

img_t.requires_grad_()
output = model(img_t)
boxes = output[0]['boxes']
labels = output[0]['labels']
scores = output[0]['scores']
img[:, :, ::1] = img[:, :, ::-1]
with open('label_list.txt', encoding='utf-8') as f:
    label_list = f.readlines()
for i in range(boxes.shape[0]):

    if float(scores[i]) > 0.6:
        cv2.rectangle(img, (int(boxes[i, 0]), int(boxes[i, 1])), (int(boxes[i, 2]), int(boxes[i, 3])), (255, 0, 255), 2)
        cv2.putText(img, label_list[labels[i]-1][:-1], (int(boxes[i, 0]), int(boxes[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
        if label_list[labels[i]-1][:-1] == 'car':
            print(i)

# cv2.imshow('iii', img)
# cv2.waitKey(0)

# for i in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13]:

scores[12].backward()
gradient_of_input = img_t.grad[0].data

def normalize(activations):
    # transform activations so that all the values be in range [0, 1]
    for i in range(len(activations)):
        activations[i] = activations[i] - torch.min(activations[i][:])
        activations[i] = activations[i] / torch.max(activations[i][:])
    return activations


def visualize_activations(image, activations):
    # activations = activations.permute(1, 2, 0)
    activations = normalize(activations)
    print(activations)
    activations[0] = activations[0] > 0.45
    activations[1] = activations[1] > 0.49
    activations[2] = activations[2] > 0.52
    activations = activations.permute(1, 2, 0).numpy()
    masked_image = np.multiply(image, activations)
    
    return masked_image


input_tensor = img1
print(gradient_of_input.shape)
print(input_tensor.shape)
receptive_field_mask = visualize_activations(input_tensor, gradient_of_input)
receptive_field_mask = cv2.cvtColor(receptive_field_mask, cv2.COLOR_RGB2BGR)
cv2.imshow("receiptive_field_max_activation", receptive_field_mask/255)
cv2.waitKey(0)