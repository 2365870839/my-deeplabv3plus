# -- coding: utf-8 --
import torch
import torchvision
import torch.nn as nn
from thop import profile
from modeling.backbone.resnet import ResNet101,ResNet

# Model
print('==> Building model..')
model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)

dummy_input = torch.randn(1, 3, 512, 512)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
