import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets

class Dconv4_Block(nn.Module):
    def __init__(self, inchannel, outchannel, midchannel):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannel, midchannel, 1, 1)
        self.offset = nn.Conv2d(midchannel, 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(midchannel, midchannel, 3, 1, 1)
        self.conv2 = nn.Conv2d(midchannel, midchannel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(midchannel)
        self.conv3 = nn.Conv2d(midchannel, outchannel, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)

        with torch.no_grad():
            offset = self.offset(x)
        dconv_x = self.dconv(x, offset)
        around_x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = (dconv_x + around_x)/2
        x = self.conv3(x)

        return x
        