import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 28)

    def forward(self, x):
        out = self.resnet18(x)
        return out
