import torch.nn as nn
import torch
from modeling.DSConv_pro import DSConv_pro
import torch.nn.functional as F


class DSLayer(nn.Module):
    def __init__(self, channel, reduction=16, device='cuda'):
        super(DSLayer, self).__init__()
        self.c1 = channel[0]
        self.c2 = channel[1]
        self.c3 = channel[2]
        self.channel = 768
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dsconv1 = DSConv_pro(self.c1, 256, device=device)
        self.dsconv2 = DSConv_pro(self.c2, 256, device=device)
        self.dsconv3 = DSConv_pro(self.c3, 256, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // reduction, self.channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1,x2,x3):
        x1 = self.dsconv1(x1)
        x2 = self.dsconv2(x2)
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = self.dsconv3(x3)
        x3 = F.interpolate(x3, x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1,x2,x3), dim=1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
