import torch
import numpy as np
from nets.dresnet_50 import dresnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection._utils import overwrite_eps
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.resnet import resnet50





def fasterrcnn_resnet50_fpn_def(weights, progress=True, num_classes=None, trainable_backbone_layers=None):
    desnet = dresnet50()
    desnet = _resnet_fpn_extractor(desnet, trainable_backbone_layers)
    model = FasterRCNN(desnet, num_classes=num_classes)
    model_dict = model.state_dict()
    pretrain_dict = torch.load(weights)
    no_load_key, temp_dict = [], {}
    for k, v in pretrain_dict.items():

        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
        else:
            no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    overwrite_eps(model, 0.0)

    return model

def fasterrcnn_resnet50_fpn(weights, progress=True, num_classes=None, trainable_backbone_layers=None):

    norm_layer = misc_nn_ops.FrozenBatchNorm2d
    resnet = resnet50(weights=None, progress=progress, norm_layer=norm_layer)
    resnet = _resnet_fpn_extractor(resnet, trainable_backbone_layers)
    model = FasterRCNN(resnet, num_classes=num_classes)
    model_dict = model.state_dict()
    
    desnet = dresnet50()
    desnet = _resnet_fpn_extractor(desnet, trainable_backbone_layers)
    dmodel = FasterRCNN(desnet, num_classes=num_classes)
    dmodel_dict = dmodel.state_dict()

    pretrain_dict = torch.load(weights)
    no_load_key, temp_dict = [], {}
    for k, v in pretrain_dict.items():

        if k in dmodel_dict.keys() and np.shape(dmodel_dict[k]) == np.shape(v):
            temp_dict[k] = v
        else:
            no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    overwrite_eps(model, 0.0)

    return model