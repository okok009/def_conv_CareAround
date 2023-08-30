import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import datetime
from tqdm import tqdm
from utils.dataloader import FRCNNDataset
#from nets.fasterrcnn_resnet50_fpn import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_def
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from utils.fit_one_epoch import fit_one_epoch

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
    
# -----------------------------------
# training_loop
# -----------------------------------
# def training_loop(epochs, optimizer, model, lr_scheduler, iter, data_loader):
#     print('---------------start training---------------')
#     update = 0
#     for epoch in range(1, epochs + 1):
#         data_loader.reset()
#         with tqdm(total=iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
#             for i in range(iter):
#                 batch_loss = 0
#                 img, bboxes, labels = data_loader()
#                 targets = []
#                 for j in range(img.shape[0]):
#                     d = {}
#                     d['boxes'] = bboxes[j]
#                     d['labels'] = labels[j]
#                     targets.append(d)
#                 loss_dict = model(img, targets)
#                 batch_loss = sum(loss for loss in loss_dict.values())
#                 optimizer.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()

#                 pbar.set_postfix(**{'batch_loss'    : batch_loss, 
#                                     'lr'            : get_lr(optimizer)})
#                 pbar.update(1)

#         lr_scheduler.step()

#         print('\n{} Epoch {}, Training loss {}\n'.format(datetime.datetime.now(), epoch, batch_loss))


# -----------------------------------
# model
# -----------------------------------
weights = './checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
# model = fasterrcnn_resnet50_fpn_def(weights, num_classes=91, trainable_backbone_layers=5)
# for k, v in model.named_parameters():
#     print(k)

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights, num_classes=91, trainable_backbone_layers=5)

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
model = model.to(device=device)

# -----------------------------------
# optimizer
# -----------------------------------
lr_rate = 0.001
milestones = [20, 30, 35]
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_rate)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

# -----------------------------------
# data_loader
# -----------------------------------
train_info_path = './train2017.txt'
val_info_path = './val2017.txt'
batch_size = 5
epochs = 40
shuffle = True
# train_iter = 2*len(train_lines)//batch_size
train_iter = 2000
val_iter = 1000
with open(train_info_path, encoding='utf-8') as f:
    train_lines = f.readlines()[0:batch_size*train_iter]
with open(val_info_path, encoding='utf-8') as f:
    val_lines = f.readlines()
train_data_loader = FRCNNDataset(train_lines, batch_size, shuffle, device)
val_data_loader = FRCNNDataset(val_lines, 1, False, device)

# -----------------------------------
# fit one epoch (train & validation)
# -----------------------------------
# training_loop(epochs, optimizer, model, lr_scheduler, iter, train_data_loader)
for epoch in range(1, epochs+1):
    fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader)