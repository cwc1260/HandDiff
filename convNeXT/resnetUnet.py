import math
import torch
import torch.nn as nn
from convNeXT.convnext import ConvNeXt, LayerNorm
from model.hourglass import Residual

BN_MOMENTUM = 0.1



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

model_list = {
          'tiny': ([3, 3, 9, 3], [96, 192, 384, 768]),
          'small': ([3, 3, 27, 3], [96, 192, 384, 768]),
          'base': ([3, 3, 27, 3], [128, 256, 512, 1024]),
          'large': ([3, 3, 27, 3], [192, 384, 768, 1536])
          }
weight_url_1k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224.pth"
}

weight_url_22k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth"
}

class convNeXTUnet(nn.Module):
    # def __init__(self, net, joint_num, pretrain, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
    def __init__(self, net, pretrain, deconv_dim=128):
        super(convNeXTUnet, self).__init__()
        # self.joint_num = joint_num
        # self.feature_dim = [self.joint_num * 3, self.joint_num]
        self.net_type = net.split('-')[-1]
        self.depths, self.dims = model_list[self.net_type]
        if pretrain == '1k':
            self.backbone = ConvNeXt(depths=self.depths, dims=self.dims, num_classes=1000)
        if pretrain == '22k':
            self.backbone = ConvNeXt(depths=self.depths, dims=self.dims, num_classes=21841)

        self.skip_layer4 = Residual(self.dims[2], self.dims[2])
        self.up4 = nn.Sequential(Residual(self.dims[3], self.dims[3]),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual(self.dims[2]+self.dims[3], self.dims[2])

        self.skip_layer3 = Residual(self.dims[1], self.dims[1])
        self.up3 = nn.Sequential(Residual(self.dims[2], self.dims[2]),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((self.dims[2]+self.dims[1]), self.dims[1])

        self.skip_layer2 = Residual(self.dims[0], self.dims[0])
        self.up2 = nn.Sequential(Residual(self.dims[1], self.dims[1]),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((self.dims[1]+self.dims[0]), deconv_dim)


        # self.feat_emb = Residual(deconv_dim, deconv_dim)
        self.result_emb = Residual(deconv_dim, deconv_dim)

        # self.finals = nn.ModuleList()
        # for out_dim in out_dim_list:
        #     self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain != '':
            if pretrain == '1k':
                url = weight_url_1k[self.net_type]
            if pretrain == '22k':
                url = weight_url_22k[self.net_type]
            print('load pre-train weight from: '+url)
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
            self.backbone.load_state_dict(checkpoint["model"])
        self.backbone.downsample_layers[0] = nn.Sequential(
            nn.Conv2d(1, self.dims[0], kernel_size=4, stride=4),
            # nn.Conv2d(1, self.dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        )

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

        # for m in self.finals.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c1, c2, c3, c4 = self.backbone.forward_features(img)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        pcl_feature = self.result_emb(img_feature)
        # result_feature = self.result_emb(img_feature)
        # img_result = torch.Tensor().to(device)
        # for layer in self.finals:
        #     temp = layer(result_feature)
        #     img_result = torch.cat((img_result, temp), dim=1)

        return pcl_feature, c4

class convNeXTUnetBig(nn.Module):
    # def __init__(self, net, joint_num, pretrain, deconv_dim=128, out_dim_list=[3*21, 21, 21]):
    def __init__(self, net, pretrain, deconv_dim=128):
        super(convNeXTUnetBig, self).__init__()
        # self.joint_num = joint_num
        # self.feature_dim = [self.joint_num * 3, self.joint_num]
        self.net_type = net.split('-')[-1]
        self.depths, self.dims = model_list[self.net_type]
        if pretrain == '1k':
            self.backbone = ConvNeXt(depths=self.depths, dims=self.dims, num_classes=1000)
        if pretrain == '22k':
            self.backbone = ConvNeXt(depths=self.depths, dims=self.dims, num_classes=21841)

        self.skip_layer4 = Residual(self.dims[2], self.dims[2])
        self.up4 = nn.Sequential(Residual(self.dims[3], self.dims[3]),
                                 nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer4 = Residual(self.dims[2]+self.dims[3], self.dims[2])

        self.skip_layer3 = Residual(self.dims[1], self.dims[1])
        self.up3 = nn.Sequential(Residual(self.dims[2], self.dims[2]),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer3 = Residual((self.dims[2]+self.dims[1]), self.dims[1])

        self.skip_layer2 = Residual(self.dims[0], self.dims[0])
        self.up2 = nn.Sequential(Residual(self.dims[1], self.dims[1]),
                                  nn.Upsample(scale_factor=2, mode='bilinear'))
        self.fusion_layer2 = Residual((self.dims[1]+self.dims[0]), deconv_dim)


        # self.feat_emb = Residual(deconv_dim, deconv_dim)
        self.result_emb = Residual(deconv_dim, deconv_dim)

        # self.finals = nn.ModuleList()
        # for out_dim in out_dim_list:
        #     self.finals.append(nn.Conv2d(in_channels=deconv_dim, out_channels=out_dim, kernel_size=1, stride=1))

        self.init_weights()
        if pretrain != '':
            if pretrain == '1k':
                url = weight_url_1k[self.net_type]
            if pretrain == '22k':
                url = weight_url_22k[self.net_type]
            print('load pre-train weight from: '+url)
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
            self.backbone.load_state_dict(checkpoint["model"])
        self.backbone.downsample_layers[0] = nn.Sequential(
            nn.Conv2d(1, self.dims[0], kernel_size=4, stride=2, padding=1),
            # nn.Conv2d(1, self.dims[0], kernel_size=4, stride=4),
            # nn.Conv2d(1, self.dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first")
        )

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

        # for m in self.finals.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, img):
        device = img.device
        c1, c2, c3, c4 = self.backbone.forward_features(img)
        # print(c1.shape, c2.shape, c3.shape, c4.shape)

        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        c3_fusion = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        c3_up = self.up3(c3_fusion)
        c2_skip = self.skip_layer3(c2)
        c2_fusion = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))

        c2_up = self.up2(c2_fusion)
        c1_skip = self.skip_layer2(c1)
        img_feature = self.fusion_layer2(torch.cat((c2_up, c1_skip), dim=1))

        pcl_feature = self.result_emb(img_feature)
        # result_feature = self.result_emb(img_feature)
        # img_result = torch.Tensor().to(device)
        # for layer in self.finals:
        #     temp = layer(result_feature)
        #     img_result = torch.cat((img_result, temp), dim=1)

        return pcl_feature, c4


if __name__ == '__main__':
    input_img = torch.rand([2, 1, 176, 176])
    model = convNeXTUnet('convnext-tiny', 21, pretrain='1k', deconv_dim=128)
    model(input_img)