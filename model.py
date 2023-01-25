"""Code for constructing the model and get the outputs from the model."""

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Function
from layers import *
import numpy as np


class RobustMseg(nn.Module):

    def __init__(self,  n_base_filters = 16, num_cls = 4, final_sigmoid=False):
        super(RobustMseg, self).__init__()
        
        self.final_sigmoid = final_sigmoid
        style_enc_list = []
        content_enc_list = []
        for i in range(4): # modality
            style_enc_list.append(Style_encoder(1, 32))
            content_enc_list.append(Content_encoder(1, 16))
            
        self.style_enc_list = nn.ModuleList(style_enc_list)
        self.content_enc_list = nn.ModuleList(content_enc_list)
        
        content_attn = []
        content_share_conv_list = []
        in_out_ch = n_base_filters
        for i in range(4): # level
            content_attn.append(BasicConv(in_out_ch*4, 4, 3, stride=1, padding=(3-1) // 2, relu=False, norm=True))
            content_share_conv_list.append(BasicConv(in_out_ch*4, in_out_ch, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True))
            in_out_ch = in_out_ch*2
            
        self.content_attn = nn.ModuleList(content_attn)
        self.content_share_conv_list = nn.ModuleList(content_share_conv_list)
        
        recon_decoder_list = []
        for i in range(4): # modality
            recon_decoder_list.append(Image_decoder())
        self.recon_decoder_list = nn.ModuleList(recon_decoder_list)
        
        self.seg_decoder = Mask_decoder(num_cls=num_cls)

    def forward(self, x, drop=None, valid=False):
        
        if drop is None:
            drop = torch.sum(x, [2,3,4]) == 0
            
        style_out = []
        content_out = [[] for _ in range(4)]
        for i in range(4): # modality
            s_out = self.style_enc_list[i](x[:, i:i+1])
#             print(s_out.shape)
            if valid:
                s_out = s_out.data.new(s_out.size()).normal_()
            style_out.append(s_out)
            content_list = self.content_enc_list[i](x[:, i:i+1])
            for j, content in enumerate(content_list):
                content_out[j].append(ZeroLayerF.apply(content, drop[:, i]))
        
        # atten
        shared_content_list = []
        for level in range(len(content_out)): # level
            share_concat = torch.cat(content_out[level], 1)
            attnmap = self.content_attn[level](share_concat)
            attnmap = F.sigmoid(attnmap)
            share_content = []
            for i in range(4):
                share_content.append(content_out[level][i] * attnmap[:, i:i+1])
            share_content = torch.cat(share_content, 1)
            share_content = self.content_share_conv_list[level](share_content)
            shared_content_list.append(share_content)
        
        out = share_content
        recon_list = []
        mu_list = []
        sigma_list = []
        for i in range(4): # modality
            recon, mu, sigma = self.recon_decoder_list[i](style_out[i], out, valid)
            recon_list.append(recon)
            mu_list.append(mu)
            sigma_list.append(sigma)
            
        recon_out = torch.cat(recon_list, 1)
        seg_out = self.seg_decoder(shared_content_list)
        if self.final_sigmoid:
            seg_out = F.sigmoid(seg_out)
        else:
            seg_out = F.softmax(seg_out, 1)
        
        return seg_out, recon_out, mu_list, sigma_list

class Style_encoder(nn.Module):

    def __init__(self, in_channels = 1, n_base_ch_se = 32):
        super(Style_encoder, self).__init__()
        
        layers = [BasicConv(in_channels, n_base_ch_se, 7, 
                                    stride=1, padding=(7-1) // 2, relu=True, norm=False)]
        
        layers.append(BasicConv(n_base_ch_se, n_base_ch_se*2, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*2, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*4, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        layers.append(BasicConv(n_base_ch_se*4, n_base_ch_se*4, 4, 
                                stride=2, padding=(4-1) // 2, relu=True, norm=False))
        self.encoder = nn.Sequential(*layers)
        self.final_conv = BasicConv(n_base_ch_se*4, n_base_ch_se*4, 1, 
                                stride=2, padding=(1-1) // 2, relu=False, norm=False)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.mean(x, [2,3,4], keepdim=True)
        x = self.final_conv(x)
        
        return x

class Content_encoder(nn.Module):

    def __init__(self, in_channels = 1, n_base_filters = 16, drop_rate = 0.3):
        super(Content_encoder, self).__init__()
        
        
        self.e1_c1 = BasicConv(in_channels, n_base_filters, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False)
        self.e1_c2 = BasicConv(n_base_filters, n_base_filters, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False, drop_rate=drop_rate)
        self.e1_c3 = BasicConv(n_base_filters, n_base_filters, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False)
        
        self.e2_c1 = BasicConv(n_base_filters, n_base_filters*2, 3, stride=2, padding=(3-1) // 2, relu=True, norm=False)
        self.e2_c2 = BasicConv(n_base_filters*2, n_base_filters*2, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False, drop_rate=drop_rate)
        self.e2_c3 = BasicConv(n_base_filters*2, n_base_filters*2, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False)
        
        
        self.e3_c1 = BasicConv(n_base_filters*2, n_base_filters*4, 3, stride=2, padding=(3-1) // 2, relu=True, norm=False)
        self.e3_c2 = BasicConv(n_base_filters*4, n_base_filters*4, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False, drop_rate=drop_rate)
        self.e3_c3 = BasicConv(n_base_filters*4, n_base_filters*4, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False)
        
        self.e4_c1 = BasicConv(n_base_filters*4, n_base_filters*8, 3, stride=2, padding=(3-1) // 2, relu=True, norm=False)
        self.e4_c2 = BasicConv(n_base_filters*8, n_base_filters*8, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False, drop_rate=drop_rate)
        self.e4_c3 = BasicConv(n_base_filters*8, n_base_filters*8, 3, stride=1, padding=(3-1) // 2, relu=True, norm=False)
        

    def forward(self, x):
        e1_x1 = self.e1_c1(x)
        e1_x2 = self.e1_c2(e1_x1)
        e1_x3 = self.e1_c3(e1_x2)
        e1_out = e1_x1 + e1_x3
        
        e2_x1 = self.e2_c1(e1_out)
        e2_x2 = self.e2_c2(e2_x1)
        e2_x3 = self.e2_c3(e2_x2)
        e2_out = e2_x1 + e2_x3
        
        e3_x1 = self.e3_c1(e2_out)
        e3_x2 = self.e3_c2(e3_x1)
        e3_x3 = self.e3_c3(e3_x2)
        e3_out = e3_x1 + e3_x3
        
        e4_x1 = self.e4_c1(e3_out)
        e4_x2 = self.e4_c2(e4_x1)
        e4_x3 = self.e4_c3(e4_x2)
        e4_out = e4_x1 + e4_x3
        
        return [e1_out, e2_out, e3_out, e4_out]


class Image_decoder(nn.Module):

    def __init__(self, in_style_ch = 128, in_content_ch = 128, mlp_ch = 128, img_ch=1):
        super(Image_decoder, self).__init__()
        channel = mlp_ch
        self.mlp = MLP(in_style_ch, mlp_ch)
        
        res_blocks = []
        for i in range(4):
            res_blocks.append(Adaptive_resblock(in_content_ch, channel))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        decoder_blocks = []
        for i in range(3):
            level_decoder = []
            level_decoder.append(nn.Upsample(scale_factor=2, mode='trilinear'))
            level_decoder.append(BasicConv(channel, channel // 2, 5, stride=1, padding=(5-1) // 2, relu=False, norm=False))
            decoder_blocks.append(nn.Sequential(*level_decoder))
            channel = channel // 2
            
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        self.final_conv = BasicConv(channel, img_ch, 7, stride=1, padding=(7-1) // 2, relu=False, norm=False)

    def forward(self, style, content, valid=False):
        mu, sigma = self.mlp(style)
        x = content
        
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x, mu, sigma)
            
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
            x = F.layer_norm(x, x.shape[1:])
            x = F.relu(x, inplace=True)
        x = self.final_conv(x)
        
        return x, mu, sigma

class Mask_decoder(nn.Module):

    def __init__(self, in_ch = 128, num_cls=4):
        super(Mask_decoder, self).__init__()
        
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d3_c1 = BasicConv(in_ch, in_ch // 2, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d3_c2 = BasicConv(in_ch, in_ch // 2, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d3_c3 = BasicConv(in_ch // 2, in_ch // 2, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True)
        
        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d2_c1 = BasicConv(in_ch // 2, in_ch // 4, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d2_c2 = BasicConv(in_ch // 2, in_ch // 4, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d2_c3 = BasicConv(in_ch // 4, in_ch // 4, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True)
        
        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.d1_c1 = BasicConv(in_ch // 4, in_ch // 8, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d1_c2 = BasicConv(in_ch // 4, in_ch // 8, 3, stride=1, padding=(3-1) // 2, relu=True, norm=True)
        self.d1_c3 = BasicConv(in_ch // 8, in_ch // 8, 1, stride=1, padding=(1-1) // 2, relu=True, norm=True)
        
        self.final_conv = BasicConv(in_ch // 8, num_cls, 1, stride=1, padding=(1-1) // 2, relu=False, norm=False)

    def forward(self, content):
        x = content[3]
        
        x = self.d3(x)
        x = self.d3_c1(x)
        x = torch.cat([x, content[2]], 1)
        x = self.d3_c2(x)
        x = self.d3_c3(x)
            
        x = self.d2(x)
        x = self.d2_c1(x)
        x = torch.cat([x, content[1]], 1)
        x = self.d2_c2(x)
        x = self.d2_c3(x)
        
        x = self.d1(x)
        x = self.d1_c1(x)
        x = torch.cat([x, content[0]], 1)
        x = self.d1_c2(x)
        x = self.d1_c3(x)
        x = self.final_conv(x)
        
        return x


class MLP(nn.Module):

    def __init__(self, in_ch = 128, mlp_ch = 128):
        super(MLP, self).__init__()
        
        self.channel = mlp_ch
        self.l1 = nn.Linear(in_ch, mlp_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(mlp_ch, mlp_ch)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.l_mu = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)

    def forward(self, style):
        x = style.view(style.size(0), -1)
        x = self.l1(x)
        x = self.relu1(x)
        
        x = self.l2(x)
        x = self.relu2(x)
        
        mu = self.l_mu(x)
        sigma = self.l_sigma(x)
        
        mu = mu.reshape(-1, self.channel, 1, 1, 1)
        sigma = sigma.reshape(-1, self.channel, 1, 1, 1)
        
        return mu, sigma 
