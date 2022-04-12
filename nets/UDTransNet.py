import torch.nn as nn
import torch
import numpy as np
from .DAT import DAT
import torchvision.models as models

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if self.scale_factor[0] > 1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

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

class DRA_C(nn.Module):
    """ Channel-wise DRA Module"""
    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                       out_channels=decoder_dim,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1,1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key =   nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,scale_factor=(self.patch_size,self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        query = self.query(decoder_L).transpose(-1, -2)
        key = self.key(trans)
        value = self.value(trans).transpose(-1, -2)
        ch_similarity_matrix = torch.matmul(query, key)
        ch_similarity_matrix = self.softmax(self.psi(ch_similarity_matrix.unsqueeze(1)).squeeze(1))
        out = torch.matmul(ch_similarity_matrix, value).transpose(-1, -2)
        out = self.out(out)
        out =  self.reconstruct(out)
        out = out * decoder_mask
        return out

class DRA_S(nn.Module):
    """ Spatial-wise DRA Module"""
    def __init__(self, skip_dim, decoder_dim, img_size, config):
        super().__init__()
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = nn.Conv2d(in_channels=decoder_dim,
                                          out_channels=decoder_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.conv = nn.Sequential(
            nn.Conv2d(decoder_dim, skip_dim, kernel_size=(1,1), bias=True),
            nn.BatchNorm2d(skip_dim),
            nn.ReLU(inplace=True))
        self.query = nn.Linear(decoder_dim, skip_dim, bias=False)
        self.key =   nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.value = nn.Linear(config.transformer.embedding_channels, skip_dim, bias=False)
        self.out = nn.Linear(skip_dim, skip_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.psi = nn.InstanceNorm2d(1)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1,scale_factor=(self.patch_size,self.patch_size))

    def forward(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder).flatten(2).transpose(-1, -2)
        query = self.query(decoder_L)
        key = self.key(trans).transpose(-1, -2)
        value = self.value(trans)
        sp_similarity_matrix = torch.matmul(query, key)
        sp_similarity_matrix = self.softmax(self.psi(sp_similarity_matrix.unsqueeze(0)).squeeze(0))
        out = torch.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out =  self.reconstruct(out)
        out = out * decoder_mask
        return out

class Up_Block(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config):
        super().__init__()
        self.scale_factor = (img_size // 14, img_size // 14)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True))
        self.pam = DRA_C(skip_ch, in_ch//2, img_size, config) # # channel_wise_DRA
        # self.pam = DRA_S(skip_ch, in_ch//2, img_size, config) # spatial_wise_DRA
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2+skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, decoder, o_i):
        d_i = self.up(decoder)
        o_hat_i = self.pam(d_i, o_i)
        x = torch.cat((o_hat_i, d_i), dim=1)
        x = self.conv(x)
        return x

class UDTransNet(nn.Module):

    def __init__(self, config, n_channels=3, n_classes=1,img_size=224):
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet34(pretrained=True)
        filters_resnet = [64,64,128,256,512]
        filters_decoder = config.decoder_channels

        # =====================================================
        # Encoder
        # =====================================================
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

        # =====================================================
        # DAT Module
        # =====================================================
        self.mtc = DAT(config, img_size, channel_num=filters_resnet[0:4], patchSize=config.patch_sizes)

        # =====================================================
        # DRA & Decoder
        # =====================================================
        self.Up5 = Up_Block(filters_resnet[4],  filters_resnet[3], filters_decoder[3], 28, config)
        self.Up4 = Up_Block(filters_decoder[3], filters_resnet[2], filters_decoder[2], 56, config)
        self.Up3 = Up_Block(filters_decoder[2], filters_resnet[1], filters_decoder[1], 112, config)
        self.Up2 = Up_Block(filters_decoder[1], filters_resnet[0], filters_decoder[0], 224, config)

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
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        o1,o2,o3,o4 = self.mtc(e1,e2,e3,e4)

        d4 = self.Up5(e5, o4)
        d3 = self.Up4(d4, o3)
        d2 = self.Up3(d3, o2)
        d1 = self.Up2(d2, o1)

        if self.n_classes ==1:
            out = self.last_activation(self.pred(d1))
        else:
            out = self.pred(d1) # if using BCEWithLogitsLoss
        return out





