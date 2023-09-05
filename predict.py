import torch
import cv2
from torchvision import transforms
from nets.fasterrcnn_resnet50_fpn import fasterrcnn_resnet50_fpn_def, fasterrcnn_resnet50_fpn_predict
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn


# weights = 'COCODevKit/checkpoints/ep40-loss1.2613470554351807.pth'
# model = fasterrcnn_resnet50_fpn_predict(weights)
# model.eval()

# weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
# model = fasterrcnn_resnet50_fpn(weights)
# model.eval()

weights = 'COCODevKit/checkpoints/ep40-loss0.8746699690818787.pth'
model = fasterrcnn_resnet50_fpn_def(weights)
model.eval()

img = 'img/street.jpg'
img = cv2.imread(img)

img[:, :, ::1] = img[:, :, ::-1]

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t = torch.unsqueeze(img_t, dim=0)

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
cv2.imshow('iii', img)
cv2.waitKey(0)