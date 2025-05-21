import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
import Models.Conv3dReLU as Conv3dReLU

import Models.LWSA as LWSA
import Models.LWCA as LWCA
import utils.configs_TransMatch as configs
import Models.Decoder as Decoder



# Encoder/Decoder
########################################################

class Conv_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm3d(out_channels)

        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm3d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out

########################################################
# Networks
########################################################
class Dual_Fusion_Attention_Net(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1 = cross_modality_attention_fusion_block(96,window_size)
        self.fusion2=cross_modality_attention_fusion_block(192,window_size)
        self.fusion3=cross_modality_attention_fusion_block(384,window_size)
        self.fusion4=cross_modality_attention_fusion_block(768,window_size)

        self.reg_head1 = Decoder.RegistrationHead(
            in_channels=96,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head2 = Decoder.RegistrationHead(
            in_channels=192,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head3 = Decoder.RegistrationHead(
            in_channels=384,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head4 = Decoder.RegistrationHead(
            in_channels=768,
            out_channels=3,
            kernel_size=3,
        )

    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)

        moved4 = self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)
        moved3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3)
        fieid3 = self.reg_head3(moved3)
        fieid3 = fieid3 + fieid4_up
        fieid3_up = self.ResizeTransformer(fieid3)

        I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        moved2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2)
        fieid2 = self.reg_head2(moved2)
        fieid2 = fieid2 + fieid3_up
        fieid2_up = self.ResizeTransformer(fieid2)

        I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)
        moved1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1)
        fieid1 = self.reg_head1(moved1)
        fieid1 = fieid1 + fieid2_up
        field1_up = self.ResizeTransformer(fieid1)
        field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1, field1_up,field1_up_up],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }


class Dual_Fusion_Attention_Net2(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net2, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1 = cross_modality_attention_block(96,window_size)
        self.fusion2=cross_modality_attention_block(192,window_size)
        self.fusion3=cross_modality_attention_block(384,window_size)
        self.fusion4=cross_modality_attention_block(768,window_size)




    def fusion(self,a,b):
        return (a+b)/2
    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)

        fieid4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)
        fieid3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = fieid3 + fieid4_up
        fieid3_up = self.ResizeTransformer(fieid3)

        I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        fieid2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = fieid2 + fieid3_up
        fieid2_up = self.ResizeTransformer(fieid2)

        I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)
        fieid1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = fieid1 + fieid2_up
        field1_up = self.ResizeTransformer(fieid1)
        field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1, field1_up,field1_up_up],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }

class Dual_Fusion_Attention_Net3(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net3, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1 = cross_modality_attention_block(96,window_size)
        self.fusion2=cross_modality_attention_block(192,window_size)
        self.fusion3=cross_modality_attention_block(384,window_size)
        self.fusion4=cross_modality_attention_block(768,window_size)




    def fusion(self,a,b):
        return (a+b)/2
    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)

        fieid4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)
        fieid3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = (fieid3 + fieid4_up)/2
        fieid3_up = self.ResizeTransformer(fieid3)

        I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        fieid2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = (fieid2 + fieid3_up)/2
        fieid2_up = self.ResizeTransformer(fieid2)

        I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)
        fieid1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = (fieid1 + fieid2_up)/2
        field1_up = self.ResizeTransformer(fieid1)
        field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4,fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1, field1_up,field1_up_up],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }


class Dual_Fusion_Attention_Net4(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net4, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1=cross_modality_attention_block(24,window_size)
        self.fusion2=cross_modality_attention_block(48,window_size)
        self.fusion3=cross_modality_attention_block(96,window_size)
        self.fusion4=cross_modality_attention_block(192,window_size)
        #################
        self.deconv_moving_img4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_moving_mask4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_fixed_img4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_fixed_mask4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        ###########
        self.deconv_moving_img3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_moving_mask3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_fixed_img3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_fixed_mask3 = nn.Sequential(

            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )

        ################
        self.deconv_moving_img2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_moving_mask2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_fixed_img2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_fixed_mask2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        ##################
        self.deconv_moving_img1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_moving_mask1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_fixed_img1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_fixed_mask1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )





    def fusion(self,a,b):
        return (a+b)/2
    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)

        I_m_feature4 = self.deconv_moving_img4(I_m_feature4)
        I_f_feature4 = self.deconv_fixed_img4(I_f_feature4)
        S_m_feature4 = self.deconv_moving_mask4(S_m_feature4)
        S_f_feature4 = self.deconv_fixed_mask4(S_f_feature4)
        fieid4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_m_feature3 = self.deconv_moving_img3(I_m_feature3)
        I_f_feature3 = self.deconv_fixed_img3(I_f_feature3)
        S_m_feature3 = self.deconv_moving_mask3(S_m_feature3)
        S_f_feature3 = self.deconv_fixed_mask3(S_f_feature3)


        I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)
        fieid3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = (fieid3 + fieid4_up)/2
        fieid3_up = self.ResizeTransformer(fieid3)

        I_m_feature2 = self.deconv_moving_img2(I_m_feature2)
        I_f_feature2 = self.deconv_fixed_img2(I_f_feature2)
        S_m_feature2 = self.deconv_moving_mask2(S_m_feature2)
        S_f_feature2 = self.deconv_fixed_mask2(S_f_feature2)

        I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        fieid2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = (fieid2 + fieid3_up)/2
        fieid2_up = self.ResizeTransformer(fieid2)

        I_m_feature1 = self.deconv_moving_img1(I_m_feature1)
        I_f_feature1 = self.deconv_fixed_img1(I_f_feature1)
        S_m_feature1 = self.deconv_moving_mask1(S_m_feature1)
        S_f_feature1 = self.deconv_fixed_mask1(S_f_feature1)

        I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)
        fieid1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = (fieid1 + fieid2_up)/2
        # field1_up = self.ResizeTransformer(fieid1)
        # field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4,fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }


class Dual_Fusion_Attention_Net5(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net5, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1=cross_modality_attention_block(24,window_size)
        self.fusion2=cross_modality_attention_block(48,window_size)
        self.fusion3=cross_modality_attention_block(96,window_size)
        self.fusion4=cross_modality_attention_block(192,window_size)
        #################
        self.deconv_moving_img4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_moving_mask4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_fixed_img4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        self.deconv_fixed_mask4 = nn.Sequential(
            Deconv3DBlock(768, 384),
            Deconv3DBlock(384, 192)
        )
        ###########
        self.deconv_moving_img3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_moving_mask3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_fixed_img3 = nn.Sequential(
            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )
        self.deconv_fixed_mask3 = nn.Sequential(

            Deconv3DBlock(384, 192),
            Deconv3DBlock(192, 96)
        )

        ################
        self.deconv_moving_img2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_moving_mask2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_fixed_img2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        self.deconv_fixed_mask2 = nn.Sequential(
            Deconv3DBlock(192, 96),
            Deconv3DBlock(96, 48)
        )
        ##################
        self.deconv_moving_img1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_moving_mask1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_fixed_img1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )
        self.deconv_fixed_mask1 = nn.Sequential(
            Deconv3DBlock(96, 48),
            Deconv3DBlock(48, 24)
        )





    def fusion(self,a,b):
        return (a+b)/2
    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)

        I_m_feature4 = self.deconv_moving_img4(I_m_feature4)
        I_f_feature4 = self.deconv_fixed_img4(I_f_feature4)
        S_m_feature4 = self.deconv_moving_mask4(S_m_feature4)
        S_f_feature4 = self.deconv_fixed_mask4(S_f_feature4)
        fieid4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_m_feature3 = self.deconv_moving_img3(I_m_feature3)
        I_f_feature3 = self.deconv_fixed_img3(I_f_feature3)
        S_m_feature3 = self.deconv_moving_mask3(S_m_feature3)
        S_f_feature3 = self.deconv_fixed_mask3(S_f_feature3)


        I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)
        fieid3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = (fieid3 + fieid4_up)/2
        fieid3_up = self.ResizeTransformer(fieid3)

        I_m_feature2 = self.deconv_moving_img2(I_m_feature2)
        I_f_feature2 = self.deconv_fixed_img2(I_f_feature2)
        S_m_feature2 = self.deconv_moving_mask2(S_m_feature2)
        S_f_feature2 = self.deconv_fixed_mask2(S_f_feature2)

        I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        fieid2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = (fieid2 + fieid3_up)/2
        fieid2_up = self.ResizeTransformer(fieid2)

        I_m_feature1 = self.deconv_moving_img1(I_m_feature1)
        I_f_feature1 = self.deconv_fixed_img1(I_f_feature1)
        S_m_feature1 = self.deconv_moving_mask1(S_m_feature1)
        S_f_feature1 = self.deconv_fixed_mask1(S_f_feature1)

        I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)
        fieid1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = (fieid1 + fieid2_up)/2
        # field1_up = self.ResizeTransformer(fieid1)
        # field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4,fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }

class Dual_Fusion_Attention_Net6(nn.Module):
    def __init__(self,window_size = (5,6,5)):
        super(Dual_Fusion_Attention_Net6, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = LWSA.LWSA(config2)
        self.fixed_lwsa_img = LWSA.LWSA(config2)
        self.moving_lwsa_seg = LWSA.LWSA(config2)
        self.fixed_lwsa_seg = LWSA.LWSA(config2)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=768)
        self.fusion1=double_cross_modality_attention_block(24*4,window_size)
        self.fusion2=double_cross_modality_attention_block(48*4,window_size)
        self.fusion3=double_cross_modality_attention_block(96*4,window_size)
        self.fusion4=double_cross_modality_attention_block(192*4,window_size)

        self.up4 = Decoder.DecoderBlock(3, 3, skip_channels=0, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)


        fieid4, I_cross_4,S_cross_4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_cross_4 = self.up(I_cross_4)
        S_cross_4 = self.up(S_cross_4)
        # I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        # S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)

        # I_moved_feature3 = torch.cat((I_m_feature3,I_cross_4),dim=1)
        # S_moved_feature3 = torch.cat((S_m_feature3,S_cross_4),dim=1)

        I_moved_feature3 = I_m_feature3
        S_moved_feature3 = S_m_feature3

        fieid3, I_cross_3, S_cross_3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3,I_cross_4,S_cross_4)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = fieid3 + fieid4_up
        fieid3_up = self.ResizeTransformer(fieid3)

        I_cross_3 = self.up(I_cross_3)
        S_cross_3 = self.up(S_cross_3)
        # I_m_feature2 = self.deconv_moving_img2(I_m_feature2)
        # I_f_feature2 = self.deconv_fixed_img2(I_f_feature2)
        # S_m_feature2 = self.deconv_moving_mask2(S_m_feature2)
        # S_f_feature2 = self.deconv_fixed_mask2(S_f_feature2)

        # I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        # S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        # I_moved_feature2 = torch.cat((I_m_feature2, I_cross_3), dim=1)
        # S_moved_feature2 = torch.cat((S_m_feature2, S_cross_3), dim=1)
        I_moved_feature2 = I_m_feature2
        S_moved_feature2 = S_m_feature2
        fieid2, I_cross_2,S_cross_2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2,I_cross_3,S_cross_3)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = fieid2 + fieid3_up
        fieid2_up = self.ResizeTransformer(fieid2)
        I_cross_2 = self.up(I_cross_2)
        S_cross_2 = self.up(S_cross_2)
        # I_m_feature1 = self.deconv_moving_img1(I_m_feature1)
        # I_f_feature1 = self.deconv_fixed_img1(I_f_feature1)
        # S_m_feature1 = self.deconv_moving_mask1(S_m_feature1)
        # S_f_feature1 = self.deconv_fixed_mask1(S_f_feature1)

        # I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        # S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)

        I_moved_feature1 = I_m_feature1
        S_moved_feature1 = S_m_feature1
        # I_moved_feature1 = torch.cat((I_m_feature1, I_cross_2), dim=1)
        # S_moved_feature1 = torch.cat((S_m_feature1, S_cross_2), dim=1)
        fieid1, I_cross_1,S_cross_1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1,I_cross_2,S_cross_2)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = fieid1 + fieid2_up

        fieid0 = self.up4(fieid1)
        field = self.up(fieid0)
        # field1_up = self.ResizeTransformer(fieid1)
        # field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4,fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1, fieid0, field],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }

from torch.cuda.amp import autocast
class CNN_Attention_Net(nn.Module):
    def __init__(self,window_size = (5,5,5)):
        super(CNN_Attention_Net, self).__init__()
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        # self.c1 = Conv3dReLU.Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        # self.c2 = Conv3dReLU.Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')

        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        self.moving_lwsa_img = Conv_encoder(1,8)
        self.fixed_lwsa_img = Conv_encoder(1,8)
        self.moving_lwsa_seg = Conv_encoder(1,8)
        self.fixed_lwsa_seg = Conv_encoder(1,8)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=8)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=16)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=32)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=64)
        self.fusion1=double_cross_modality_attention_block2(8,window_size)
        self.fusion2=double_cross_modality_attention_block2(16,window_size)
        self.fusion3=double_cross_modality_attention_block2(32,window_size)
        self.fusion4=double_cross_modality_attention_block2(64,window_size)

        self.up4 = Decoder.DecoderBlock(3, 3, skip_channels=0, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    @autocast()
    def forward(self,I_m,I_f,S_m,S_f):
        I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = self.moving_lwsa_img(I_m)
        I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = self.fixed_lwsa_img(I_f)
        S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = self.moving_lwsa_seg(S_m)
        S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = self.fixed_lwsa_seg(S_f)


        fieid4, I_cross_4,S_cross_4= self.fusion4(I_m_feature4, I_f_feature4, S_m_feature4, S_f_feature4)
        # fieid4 = self.reg_head4(moved4)
        fieid4_up = self.ResizeTransformer(fieid4)

        I_cross_4 = self.up(I_cross_4)
        S_cross_4 = self.up(S_cross_4)
        # I_moved_feature3 = self.SpatialTransformer(I_m_feature3, fieid4_up)
        # S_moved_feature3 = self.SpatialTransformer(S_m_feature3, fieid4_up)

        # I_moved_feature3 = torch.cat((I_m_feature3,I_cross_4),dim=1)
        # S_moved_feature3 = torch.cat((S_m_feature3,S_cross_4),dim=1)

        I_moved_feature3 = I_m_feature3
        S_moved_feature3 = S_m_feature3

        fieid3, I_cross_3, S_cross_3 = self.fusion3(I_moved_feature3,I_f_feature3,S_moved_feature3,S_f_feature3,I_cross_4,S_cross_4)
        # fieid3 = self.reg_head3(moved3)
        fieid3 = fieid3 + fieid4_up
        fieid3_up = self.ResizeTransformer(fieid3)

        I_cross_3 = self.up(I_cross_3)
        S_cross_3 = self.up(S_cross_3)
        # I_m_feature2 = self.deconv_moving_img2(I_m_feature2)
        # I_f_feature2 = self.deconv_fixed_img2(I_f_feature2)
        # S_m_feature2 = self.deconv_moving_mask2(S_m_feature2)
        # S_f_feature2 = self.deconv_fixed_mask2(S_f_feature2)

        # I_moved_feature2 = self.SpatialTransformer(I_m_feature2, fieid3_up)
        # S_moved_feature2 = self.SpatialTransformer(S_m_feature2, fieid3_up)
        # I_moved_feature2 = torch.cat((I_m_feature2, I_cross_3), dim=1)
        # S_moved_feature2 = torch.cat((S_m_feature2, S_cross_3), dim=1)
        I_moved_feature2 = I_m_feature2
        S_moved_feature2 = S_m_feature2
        fieid2, I_cross_2,S_cross_2 = self.fusion2(I_moved_feature2,I_f_feature2,S_moved_feature2,S_f_feature2,I_cross_3,S_cross_3)
        # fieid2 = self.reg_head2(moved2)
        fieid2 = fieid2 + fieid3_up
        fieid2_up = self.ResizeTransformer(fieid2)
        I_cross_2 = self.up(I_cross_2)
        S_cross_2 = self.up(S_cross_2)
        # I_m_feature1 = self.deconv_moving_img1(I_m_feature1)
        # I_f_feature1 = self.deconv_fixed_img1(I_f_feature1)
        # S_m_feature1 = self.deconv_moving_mask1(S_m_feature1)
        # S_f_feature1 = self.deconv_fixed_mask1(S_f_feature1)

        # I_moved_feature1 = self.SpatialTransformer(I_m_feature1, fieid2_up)
        # S_moved_feature1 = self.SpatialTransformer(S_m_feature1, fieid2_up)

        I_moved_feature1 = I_m_feature1
        S_moved_feature1 = S_m_feature1
        # I_moved_feature1 = torch.cat((I_m_feature1, I_cross_2), dim=1)
        # S_moved_feature1 = torch.cat((S_m_feature1, S_cross_2), dim=1)
        fieid1, I_cross_1,S_cross_1 = self.fusion1(I_moved_feature1,I_f_feature1,S_moved_feature1,S_f_feature1,I_cross_2,S_cross_2)
        # fieid1 = self.reg_head1(moved1)
        fieid1 = fieid1 + fieid2_up

        fieid0 = self.up4(fieid1)
        field =fieid1
        # field = self.up(fieid0)
        # field1_up = self.ResizeTransformer(fieid1)
        # field1_up_up = self.ResizeTransformer(field1_up)

        return {"field":[fieid4,fieid4_up,fieid3, fieid3_up,fieid2, fieid2_up, fieid1, fieid0, field],
                 "fixed_img_feature" : [I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4],
                 "moving_img_feature" : [I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4],
                 "fixed_mask_feature" : [S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4],
                 "moving_mask_feature" : [S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4]
                }


class Boundary_image_reg(nn.Module):
    def __init__(self):
        super(Boundary_image_reg, self).__init__()
        self.img_encoder = Conv_encoder()
        self.up0 = Decoder.DecoderBlock(768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = Decoder.DecoderBlock(384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = Decoder.DecoderBlock(192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = Decoder.DecoderBlock(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = Decoder.DecoderBlock(48, 16, skip_channels=16, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)


def extract_class_boundaries(mask, num_classes):
    """
    提取每个类别的边界
    mask: 输入分割 Mask (B, 1, H, W, D)
    num_classes: Mask 中类别的总数
    返回: 每个类别的边界字典 {class_id: boundary (B, 1, H, W, D)}
    """
    boundaries = {}

    # 遍历每个类别
    for class_id in range(num_classes):
        # 创建类别特定的二值 Mask
        class_mask = (mask == class_id).float()  # 生成类别为 class_id 的二值掩码

        # 定义 Sobel 算子（3D）
        kernel_x = torch.tensor([
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        kernel_y = torch.tensor([
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        kernel_z = torch.tensor([
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 将卷积核移动到同一设备
        device = mask.device
        kernel_x = kernel_x.to(device)
        kernel_y = kernel_y.to(device)
        kernel_z = kernel_z.to(device)

        # 计算梯度
        grad_x = nnf.conv3d(class_mask, kernel_x, padding=1)
        grad_y = nnf.conv3d(class_mask, kernel_y, padding=1)
        grad_z = nnf.conv3d(class_mask, kernel_z, padding=1)

        # 计算边界强度
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

        # 将边界信息二值化
        boundaries[class_id] = (boundary > 0).float()

    return boundaries

class Conv_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm3d(out_channels)

        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm3d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out


class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            # SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class cross_attention_block(nn.Module):
    def __init__(self,dim,window_size=(5,6,5)):
        super(cross_attention_block, self).__init__()
        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size=window_size
        config2.depths = (2, 2, 2, 2)
        config2.num_heads = (1, 1, 1, 1)
        self.lwca = LWCA.LWCA(config2, dim_diy=dim)
        # self.lwsa = LWSA.LWSA2(config2)

        self.reg_head_I = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head_S = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
    def forward(self,I_m,I_f):


        I_cross_feature = self.lwca(I_m,I_f)
        # S_cross_feature = self.lwca(S_m,S_f)
        I_out= self.reg_head_I(I_cross_feature)
        # S_out= self.reg_head_S(S_cross_feature)
        # out = (I_out+S_out)/2
        # return out
        return I_out


class cross_modality_attention_block(nn.Module):
    def __init__(self,dim,window_size=(5,6,5)):
        super(cross_modality_attention_block, self).__init__()
        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size=window_size
        config2.depths = (2, 2, 2, 2)
        config2.num_heads = (1, 1, 1, 1)
        self.lwca = LWCA.LWCA(config2, dim_diy=dim)
        # self.lwsa = LWSA.LWSA2(config2)

        self.reg_head_I = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head_S = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
    def forward(self,I_m,I_f,S_m,S_f):


        I_cross_feature = self.lwca(I_m,I_f)
        S_cross_feature = self.lwca(S_m,S_f)
        I_out= self.reg_head_I(I_cross_feature)
        S_out= self.reg_head_S(S_cross_feature)
        out = (I_out+S_out)/2
        return out

class double_cross_modality_attention_block(nn.Module):
    def __init__(self,dim,window_size=(5,6,5)):
        super(double_cross_modality_attention_block, self).__init__()
        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size=window_size
        config2.depths = (2, 2, 2, 2)
        config2.num_heads = (1, 1, 1, 1)
        self.lwca = LWCA.LWCA(config2, dim_diy=dim)
        # self.lwsa = LWSA.LWSA2(config2)
        self.ConvReLU_img = Conv3dReLU.Conv3dReLU(dim*2,dim,3, 1, use_batchnorm=False)
        self.ConvReLU_seg = Conv3dReLU.Conv3dReLU(dim*2,dim,3, 1, use_batchnorm=False)

        self.ConvReLU_img2 = Conv3dReLU.Conv3dReLU(dim * 4, dim, 3, 1, use_batchnorm=False)
        self.ConvReLU_seg2 = Conv3dReLU.Conv3dReLU(dim * 4, dim, 3, 1, use_batchnorm=False)
        self.reg_head_I = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head_S = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
    def forward(self,I_m,I_f,S_m,S_f,pre_I=None,pre_S=None):


        I_cross_feature1 = self.ConvReLU_img(torch.cat((I_m,I_f),dim=1))
        I_cross_feature2 = self.ConvReLU_img(I_f,I_m)

        I_cross_feature1 = self.lwca(I_m, I_f)
        I_cross_feature2 = self.lwca(I_f, I_m)
        I_cross_feature = torch.cat((I_cross_feature1,I_cross_feature2),dim=1)
        if pre_I is not None:
            I_cross_feature = torch.cat((I_cross_feature,pre_I),dim=1)
            I_cross_feature = self.ConvReLU_img2(I_cross_feature)

        else:
            I_cross_feature = self.ConvReLU_img(I_cross_feature)

        S_cross_feature1 = self.lwca(S_m,S_f)
        S_cross_feature2 = self.lwca(S_m,S_f)
        S_cross_feature = torch.cat((S_cross_feature1,S_cross_feature2),dim=1)
        if pre_S is not None:
            S_cross_feature = torch.cat((S_cross_feature,pre_S),dim=1)
            S_cross_feature = self.ConvReLU_img2(S_cross_feature)

        else:
            S_cross_feature = self.ConvReLU_img(S_cross_feature)
        # S_cross_feature = self.ConvReLU_img(S_cross_feature)

        I_out= self.reg_head_I(I_cross_feature)
        S_out= self.reg_head_S(S_cross_feature)
        # out = S_out

        out = (I_out+S_out)/2
        return out,I_cross_feature,S_cross_feature


class double_cross_modality_attention_block2(nn.Module):
    def __init__(self, dim, window_size=(5, 5, 5)):
        super(double_cross_modality_attention_block2, self).__init__()
        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size = window_size
        config2.depths = (2, 2, 2, 2)
        config2.num_heads = (1, 1, 1, 1)
        self.lwca = LWCA.LWCA(config2, dim_diy=dim)
        # self.lwsa = LWSA.LWSA2(config2)
        self.ConvReLU_img = Conv3dReLU.Conv3dReLU(dim * 2, dim, 3, 1, use_batchnorm=False)
        self.ConvReLU_seg = Conv3dReLU.Conv3dReLU(dim * 2, dim, 3, 1, use_batchnorm=False)

        self.ConvReLU_img2 = Conv3dReLU.Conv3dReLU(dim * 4, dim, 3, 1, use_batchnorm=False)
        self.ConvReLU_seg2 = Conv3dReLU.Conv3dReLU(dim * 4, dim, 3, 1, use_batchnorm=False)
        self.reg_head_I = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )
        self.reg_head_S = Decoder.RegistrationHead(
            in_channels=dim,
            out_channels=3,
            kernel_size=3,
        )

    def forward(self, I_m, I_f, S_m, S_f, pre_I=None, pre_S=None):

        I_cross_feature = self.ConvReLU_img(torch.cat((I_m, I_f), dim=1))
        S_cross_feature = self.ConvReLU_img(torch.cat((S_m, S_f), dim=1))





        # S_cross_feature = self.ConvReLU_img(S_cross_feature)

        I_out = self.reg_head_I(I_cross_feature)
        S_out = self.reg_head_S(S_cross_feature)
        # out = S_out

        out = (I_out + S_out) / 2
        return out, I_cross_feature, S_cross_feature


class cross_modality_attention_fusion_block(nn.Module):
    def __init__(self,dim,window_size=(5,6,5)):
        super(cross_modality_attention_fusion_block, self).__init__()
        config2 = configs.get_TransMatch_LPBA40_config()
        config2.window_size=window_size
        config2.depths = (2, 2, 2, 2)
        config2.num_heads = (1, 1, 1, 1)
        self.lwca = LWCA.LWCA(config2, dim_diy=dim)
        # self.lwsa = LWSA.LWSA2(config2)

    def forward(self,I_m,I_f,S_m,S_f):


        I_cross_feature = self.lwca(I_m,I_f)
        S_cross_feature = self.lwca(S_m,S_f)

        fusion_feature = self.lwca(I_cross_feature,S_cross_feature)

        return fusion_feature

class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x
