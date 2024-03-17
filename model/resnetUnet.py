import math
import torch
import torch.nn as nn
from model.resnet import BasicBlock, Bottleneck, ResNet
from model.hourglass import Residual
import torchvision

BN_MOMENTUM = 0.1

resnet = {18: (BasicBlock, [2, 2, 2, 2]),
          50: (Bottleneck, [3, 4, 6, 3]),
          101: (Bottleneck, [3, 4, 23, 3]),
          152: (Bottleneck, [3, 8, 36, 3])
          }


def conv_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=1,
                padding=1,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


def convtranspose_bn_relu(in_planes, out_planes, kernel):
    return nn.Sequential(
        nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.1),
        nn.ReLU(inplace=True)
    )


class OfficialResNetUnet(nn.Module):
    def __init__(self, backbone, joint_num, pretrain=True, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
        super(OfficialResNetUnet, self).__init__()
        self.joint_num = joint_num
        self.feature_dim = [self.joint_num * 3, self.joint_num]
        layers_num = int(backbone.split('-')[-1])
        block, layers = resnet[layers_num]
        self.backbone = ResNet(block, layers)
        self.skip_layer4 = Residual(256*block.expansion, 256)
        self.up4 = nn.Sequential(Residual(512*block.expansion, 512),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual((512+256), 256)

        self.skip_layer3 = Residual(128*block.expansion, 128)
        self.up3 = nn.Sequential(Residual(256, 256),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((256+128), 128)

        self.skip_layer2 = Residual(64*block.expansion, 64)
        self.up2 = nn.Sequential(Residual(128, 128),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((128+64), deconv_dim)

        self.finals = nn.ModuleList()
        for out_dim in out_dim_list:
            self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain:
            if layers_num == 18:
                print('load weight from resnet-18')
                pretrain_weight = torchvision.models.resnet18(pretrained=True)
            elif layers_num == 50:
                print('load weight from resnet-50')
                pretrain_weight = torchvision.models.resnet50(pretrained=True)
            elif layers_num == 101:
                print('load weight from resnet-101')
                pretrain_weight = torchvision.models.resnet101(pretrained=True)
            self.backbone.load_state_dict(pretrain_weight.state_dict(), strict=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

        for m in self.finals.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c0, c1, c2, c3, c4 = self.backbone(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        img_result = torch.Tensor().to(device)
        for layer in self.finals:
            temp = layer(img_feature)
            img_result = torch.cat((img_result, temp), dim=1)

        return img_result, img_feature
