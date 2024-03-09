import math
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.DSConv_pro import DSConv_pro
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DSC_block(nn.Module):
    def __init__(self, in_ch, out_ch, device):
        super(DSC_block, self).__init__()
        self.dsconv_x = DSConv_pro(in_ch, out_ch, morph=0, device=device)
        self.dsconv_y = DSConv_pro(in_ch, out_ch, morph=1, device=device)
        self.conv1 = nn.Conv2d(in_ch + 2 * out_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_x = self.dsconv_x(x)
        x_y = self.dsconv_y(x)
        x = torch.cat([x, x_x, x_y], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        # layers = [3, 4, 23, 3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dsc1 = DSC_block(64, 64, 'cpu')
        self.dsc2 = DSC_block(256, 256, 'cpu')
        self.dsc3 = DSC_block(512, 512, 'cpu')
        # self.dsc4 = DSC_block(1024, 1024, 'cpu')
        self.conv2 = nn.Conv2d(832, 256, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        # downsample = nn.Sequential(
        #               nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=1, stride=1, bias=False)
        #               nn.BatchNorm2d(in_channels=64*4)
        # block(inplanes=64, planes=64, stride=1, dilation=1, downsample=dowensample, BatchNorm=nn.BatchNorm2d)
        # block(inplanes=256, planes=64, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d) * 2

        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        # downsample = nn.Sequential(
        #               nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False)
        #               nn.BatchNorm2d(in_channels=512)
        # block(inplanes=256, planes=128, stride=2, dilation=1, downsample=dowensample, BatchNorm=nn.BatchNorm2d)
        # block(inplanes=512, planes=128, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d) * 3

        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        # downsample = nn.Sequential(
        #               nn.Conv2d(in_channels=512, out_channels=256*4, kernel_size=1, stride=2, bias=False)
        #               nn.BatchNorm2d(in_channels=256*4)
        # block(inplanes=512, planes=256, stride=2, dilation=1, downsample=dowensample, BatchNorm=nn.BatchNorm2d)
        # block(inplanes=1024, planes=256, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d) * 22

        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        # downsample = nn.Sequential(
        #               nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, bias=False)
        #               nn.BatchNorm2d(in_channels=2048)
        # block(inplanes=1024, planes=512, stride=1, dilation=2, downsample=dowensample, BatchNorm=nn.BatchNorm2d)
        # block(inplanes=2048, planes=512, stride=1, dilation=4, downsample=None, BatchNorm=nn.BatchNorm2d)
        # block(inplanes=2048, planes=512, stride=1, dilation=8, downsample=None, BatchNorm=nn.BatchNorm2d)

        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        # input: 1,3,512,512
        x = self.conv1(input)  # 1,64,256,256
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1,64,128,128
        x1 = self.dsc1(x)  # 1,64,128,128
        x = self.layer1(x)  # 1,256,128,128
        x2 = self.dsc2(x)  # 1,256,128,128
        x = self.layer2(x)  # 1,512,64,64
        x3 = self.dsc3(x)  # 1,512,64,64
        x3 = F.interpolate(x3, x2.size()[2:], mode='bilinear', align_corners=True)  # 1,512,128,128
        x = self.layer3(x)  # 1,1024,32,32
        x = self.layer4(x)  # 1,2048,32,32
        fused_map = torch.cat([x1, x2, x3], dim=1)  # 1,832,128,128
        fused_map = self.conv2(fused_map)  # 1,256,128,128

        return x, fused_map

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model


if __name__ == "__main__":
    import torch

    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
