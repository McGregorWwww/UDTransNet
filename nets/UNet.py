import torch.nn as nn
import torch
import torch.functional as F
import torchvision.models as models

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class Up_Block_sk(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2),
            # nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2+skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))



    def forward(self, x, skip):
        # print(x.size())
        # print(skip.size())
        x = self.up(x)
        up_sample = x
        # print(x.size())
        x = torch.cat((skip, x), dim=1)
        # print(x.size())
        x = self.conv(x)
        return x, up_sample


class Up_Block_sk_noCat(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2),
            # nn.Conv2d(in_ch, in_ch//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))



    def forward(self, x, skip):
        # print(x.size())
        # print(skip.size())
        x = self.up(x)
        # print(x.size())
        # x = torch.cat((skip, x), dim=1)
        # print(x.size())
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class UNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_channels=3, n_classes=9):
        super().__init__()
        self.n_classes = n_classes
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] #  64 128 256 512 1024
        # filters = [16,64,256,512,1024]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = Up_Block(filters[4], filters[3])
        self.Up4 = Up_Block(filters[3], filters[2])
        self.Up3 = Up_Block(filters[2], filters[1])
        self.Up2 = Up_Block(filters[1], filters[0])

        # self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.pred = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, kernel_size=1),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, n_classes, kernel_size=1),
        )



    def forward(self, x):
        # print(x.size())

        e1 = self.Conv1(x)
        # print(e1[0][0])

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5,e4)
        d4 = self.Up4(d5,e3)
        d3 = self.Up3(d4,e2)
        d2 = self.Up2(d3,e1)

        if self.n_classes == 1:
            out = nn.Sigmoid()(self.pred(d2))
        else:
            out = self.pred(d2)


        return out

class Up_Block(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))


    def forward(self, x, skip):
        # print(x.size())
        # print(skip.size())
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        # print(x.size())
        x = self.conv(x)
        return x

class Down_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down_block, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.Maxpool(x)
        x = self.conv(x)
        return x

class R50_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.n_classes = n_classes
        self.vis = vis
        resnet = models.resnet50(pretrained=True)
        filters_resnet = [64,256,512,1024,2048]

        self.Conv1 = nn.Sequential(
            nn.Conv2d(n_channels, filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_resnet[0], filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True))
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4

        self.Up5 = Up_Block(filters_resnet[4], filters_resnet[3])
        self.Up4 = Up_Block(filters_resnet[3], filters_resnet[2])
        self.Up3 = Up_Block(filters_resnet[2], filters_resnet[1])
        self.Up2 = Up_Block(filters_resnet[1], filters_resnet[0])

        self.pred = nn.Sequential(
            nn.Conv2d(filters_resnet[0], 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, n_classes, kernel_size=1),
        )
        self.last_activation = nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e1_maxp = self.Maxpool(e1)
        e2 = self.Conv2(e1_maxp)
        # print(e2.size())
        # e2_maxp = self.Maxpool(e2)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        d5 = self.Up5(e5,e4)
        d4 = self.Up4(d5,e3)
        d3 = self.Up3(d4,e2)
        d2 = self.Up2(d3,e1)

        if self.n_classes ==1:
            out = self.last_activation(self.pred(d2))
        else:
            out = self.pred(d2) # if nusing BCEWithLogitsLoss
        return out

class R34_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.n_classes = n_classes
        self.vis = vis
        resnet = models.resnet34(pretrained=False)
        filters_resnet = [64, 64,128,256,512]
        filters_decoder= [32, 64,128,256,512]

        self.Conv1 = nn.Sequential(
            nn.Conv2d(n_channels, filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_resnet[0], filters_resnet[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters_resnet[0]),
            nn.ReLU(inplace=True))
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4

        self.Up5 = Up_Block_sk(filters_resnet[4],  filters_resnet[3], filters_decoder[3])
        self.Up4 = Up_Block_sk(filters_decoder[3], filters_resnet[2], filters_decoder[2])
        self.Up3 = Up_Block_sk(filters_decoder[2], filters_resnet[1], filters_decoder[1])
        self.Up2 = Up_Block_sk(filters_decoder[1], filters_resnet[0], filters_decoder[0])

        self.pred = nn.Sequential(
            nn.Conv2d(filters_decoder[0], filters_decoder[0]//2, kernel_size=1),
            nn.BatchNorm2d(filters_decoder[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_decoder[0]//2, n_classes, kernel_size=1),
        )
        self.last_activation = nn.Sigmoid()



    def forward(self, x):
        e1 = self.Conv1(x)
        e1_maxp = self.Maxpool(e1)
        e2 = self.Conv2(e1_maxp)
        # print(e2.size())
        # e2_maxp = self.Maxpool(e2)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        d5, up5 = self.Up5(e5,e4)
        d4, up4 = self.Up4(d5,e3)
        d3, up3 = self.Up3(d4,e2)
        d2, up2 = self.Up2(d3,e1)

        if self.n_classes ==1:
            out = self.last_activation(self.pred(d2))
        else:
            out = self.pred(d2) # if nusing BCEWithLogitsLoss

        # en_feat = []
        # en_feat.append(e1)
        # en_feat.append(e2)
        # en_feat.append(e3)
        # en_feat.append(e4)
        #
        # up_feat = []
        # up_feat.append(up2)
        # up_feat.append(up3)
        # up_feat.append(up4)
        # up_feat.append(up5)
        #
        # de_feat = []
        # de_feat.append(d2)
        # de_feat.append(d3)
        # de_feat.append(d4)
        # de_feat.append(d5)

        return out

        # return out, en_feat, up_feat, de_feat
