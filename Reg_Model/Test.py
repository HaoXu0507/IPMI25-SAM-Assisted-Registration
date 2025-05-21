# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Models.STN import SpatialTransformer
from natsort import natsorted
import cv2 as cv
from Models.TransMatch import TransMatch
import json
import nibabel as nib
import torch.nn.functional as F
import random
from torch.cuda.amp import GradScaler, autocast
import skimage.measure as measure
# import torchvision.transforms as transforms
import monai.losses.dice as monai_loss
import torchio as tio
import torch.nn as nn
from Models.CorrMLP import CorrMLP
from Models.DualStream_Voxel_Contrast import Dual_Fusion_Attention_Net
from scipy.ndimage import binary_dilation, binary_erosion
import monai
from TransMorph_models.TransMorph_bspl import TranMorphBSplineNet
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes


def jacobian_determinant(disp):
    device = disp.device

    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(device)

    jacobian = torch.cat((gradz(disp), grady(disp), gradx(disp)), 0) + torch.eye(3, 3, device=device).view(3, 3, 1, 1,
                                                                                                           1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                   :, :]) + \
             jacobian[2, 0, :, :, :] * (
                     jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                   :, :])

    return jacdet.unsqueeze(0).unsqueeze(0)






def prepare_data(json_dir, split='train'):
    with open(json_dir, 'r') as f:
        lines = f.readlines()
    all_data = [json.loads(line) for line in lines]
    data = []
    if split == 'train':
        data_idx = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,]

        for i in data_idx:
            data.append(all_data[i])
    elif split == 'test':
        data_idx = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

        for i in data_idx:
            data.append(all_data[i])
    else:
        data_idx = range(30)
        for i in data_idx:
            data.append(all_data[i])

    all_img = np.zeros((len(data), 1, 160, 160, 160))  # pin_memory()
    all_mask = np.zeros((len(data), 1, 160, 160, 160))  # pin_memory()
    all_seg = np.zeros((len(data), 1, 160, 160, 160))  # pin_memory()

    for i in range(len(data)):
        data_i = data[i]
        image_dir = data_i['image']
        mask_dir = data_i['mask']
        seg_dir = data_i['seg']
        label_name = data_i['label']

        image = torch.from_numpy(nib.load(image_dir).get_fdata())
        mask = torch.from_numpy(nib.load(mask_dir).get_fdata())
        seg = torch.from_numpy(nib.load(seg_dir).get_fdata())

        ss = seg.max()
        seg = torch.where(seg == 26, torch.ones_like(seg) * 12, seg)
        seg = torch.where(seg == 27, torch.ones_like(seg) * 13, seg)
        seg = torch.where(seg > 13, torch.zeros_like(seg), seg)

        image = F.interpolate(image[None, None], (160, 160, 160), mode='trilinear')
        mask = F.interpolate(mask[None, None], (160, 160, 160), mode='nearest').long()
        seg = F.interpolate(seg[None, None], (160, 160, 160), mode='nearest').long()

        all_img[i] = image[0].detach().numpy()
        all_mask[i] = mask[0].detach().numpy()
        all_seg[i] = seg[0].detach().numpy()


    pairs = []
    for i in range(len(data)):
        for j in range(len(data)):
            if (i >= j):
                continue
            pairs.append([i, j])
    data = {'images': all_img,
            'masks': all_mask,
            'segmentations': all_seg,
            'labels': label_name,
            'pairs': pairs}

    return data


def prepare_data_ACDC(json_dir, split='train'):
    with open(json_dir, 'r') as f:
        lines = f.readlines()
    all_data = [json.loads(line) for line in lines]
    data = []
    if split == 'train':
        data_idx = range(90)
        for i in data_idx:
            data.append(all_data[i])
    elif split == 'test':
        data_idx = range(10)
        for i in data_idx:
            data.append(all_data[i])
    

    all_fixed_img = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()
    all_fixed_mask = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()
    all_fixed_seg = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()
    all_moving_img = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()
    all_moving_mask = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()
    all_moving_seg = np.zeros((len(data), 1, 128, 128, 32))  # pin_memory()

    for i in range(len(data)):
        data_i = data[i]
        fixed_image_dir = data_i['fixed_image']
        fixed_mask_dir = data_i['fixed_mask']
        fixed_seg_dir = data_i['fixed_seg']
        moving_image_dir = data_i['moving_image']
        moving_mask_dir = data_i['moving_mask']
        moving_seg_dir = data_i['moving_seg']
        label_name = data_i['label']

        fixed_image = torch.from_numpy(nib.load(fixed_image_dir).get_fdata())
        fixed_mask = torch.from_numpy(nib.load(fixed_mask_dir).get_fdata())
        fixed_seg = torch.from_numpy(nib.load(fixed_seg_dir).get_fdata())

        moving_image = torch.from_numpy(nib.load(moving_image_dir).get_fdata())
        moving_mask = torch.from_numpy(nib.load(moving_mask_dir).get_fdata())
        moving_seg = torch.from_numpy(nib.load(moving_seg_dir).get_fdata())


        fixed_seg = fixed_seg / 2
        moving_seg = moving_seg / 2
        fixed_seg = torch.where(fixed_seg > 3, torch.zeros_like(fixed_seg), fixed_seg)
        moving_seg = torch.where(moving_seg > 3, torch.zeros_like(moving_seg), moving_seg)

        moving_image = F.interpolate(moving_image[None, None], (128, 128, 32), mode='trilinear')
        moving_mask = F.interpolate(moving_mask[None, None], (128, 128, 32), mode='nearest').long()
        moving_seg = F.interpolate(moving_seg[None, None], (128, 128, 32), mode='nearest').long()

        fixed_image = F.interpolate(fixed_image[None, None], (128, 128, 32), mode='trilinear')
        fixed_mask = F.interpolate(fixed_mask[None, None], (128, 128, 32), mode='nearest').long()
        fixed_seg = F.interpolate(fixed_seg[None, None], (128, 128, 32), mode='nearest').long()


        all_fixed_img[i] = fixed_image[0].detach().numpy()
        all_fixed_mask[i] = fixed_mask[0].detach().numpy()
        all_fixed_seg[i] = fixed_seg[0].detach().numpy()
        all_moving_img[i] = moving_image[0].detach().numpy()
        all_moving_mask[i] = moving_mask[0].detach().numpy()
        all_moving_seg[i] = moving_seg[0].detach().numpy()

    data = {
        'fixed_images': all_fixed_img,
        'fixed_masks': all_fixed_mask,
        'fixed_segmentations': all_fixed_seg,
        'moving_images': all_moving_img,
        'moving_masks': all_moving_mask,
        'moving_segmentations': all_moving_seg,
        'labels': label_name,
    }

    return data




def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))



def compute_label_dice(gt, pred):

    cls_lst = range(1, 16)
    # cls_lst = [182]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


def mask2multi_class(mask, mask_max=15):
    N, H, W, D = mask.shape
    multi_mask = torch.zeros((N, mask_max, H, W, D))
    for i in range(mask_max):
        multi_mask[:, i] = mask[0] == i + 1
    return multi_mask


def multi_class2mask(mask):
    c, H, W, D = mask.shape
    out = np.zeros((H, W, D))
    for i in range(c):
        out = np.where(mask[i] == 1, np.ones((H, W, D)) * (i + 1), out)
    out = out.astype(np.float)
    return out



def train():
    # for Abdomen CT
    json_dir = '/home/hao/PycharmProjects/SAT-main/ACDC/TransMatch_TMI-main/ACDC_image_seg_data.jsonl'
    data = prepare_data_ACDC(json_dir=json_dir, split='test')
    vol_size = [160, 160, 160]

    # for ACDC MRI
    # json_dir = '/home/hao/PycharmProjects/SAT-main/TransMatch_TMI-main/segmentation_result_Abdomen_CT_CT_data.jsonl'
    # data = prepare_data(json_dir = json_dir,split='test')
    # vol_size = [128, 128, 32]

    device = torch.device('cuda')

    from Models.Conv_Transformer_Reg import, CNN_Attention_Net
    net = CNN_Attention_Net().to(device)

    ckpt = torch.load('your_ckpt.tar')
    net = ckpt['state_dict'].to(device)
    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    net.eval()
    STN.eval()
    STN_label.eval()

    N, _, H, W, D = data['images'].shape
    pairs = data['pairs']
    pair_max = len(pairs)
    index_list = np.array(range(pair_max))
    random.shuffle(index_list)
    idx = 0
    epoch_dice = 0


    ## test

    #### ACDC ####
    mask_max = test_data['fixed_masks'].max()
    seg_max = test_data['fixed_segmentations'].max()

    N, _, H, W, D = test_data['fixed_images'].shape
    pair_max = N

    index_list = np.array(range(pair_max))
    final_dice = []
    jac_det_list = []
    idx = 0
    for i in index_list:
        with torch.no_grad():
            idx += 1

            moving_img = test_data['moving_images'][i]
            moving_seg = test_data['moving_segmentations'][i]
            moving_mask = test_data['moving_masks'][i]

            fixed_img = test_data['fixed_images'][i]
            fixed_seg = test_data['fixed_segmentations'][i]
            fixed_mask = test_data['fixed_masks'][i]


            moving_img = torch.from_numpy(moving_img).cuda().unsqueeze(0)
            moving_seg = torch.from_numpy(moving_seg).cuda()
            moving_mask = torch.from_numpy(moving_mask).cuda()

            fixed_img = torch.from_numpy(fixed_img).cuda().unsqueeze(0)
            fixed_seg = torch.from_numpy(fixed_seg).cuda()
            fixed_mask = torch.from_numpy(fixed_mask).cuda()


            multi_moving_mask = mask2multi_class(moving_mask, 3).contiguous().cuda()
            multi_fixed_mask = mask2multi_class(fixed_mask, 3).contiguous().cuda()
            with torch.no_grad():

                out = net(moving_img.float(), fixed_img.float(), moving_seg.float().unsqueeze(0),
                          fixed_seg.float().unsqueeze(0),)
            #
            fieid4, fieid4_up, fieid3, fieid3_up, fieid2, fieid2_up, fieid1, fieid0, field = out['field']
            #
            flow_m2f = field
            m2f = STN(moving_img.float(), flow_m2f.float())
            m2f_mask = STN_label(multi_moving_mask, flow_m2f)
            grid = mk_grid_img().float()
            m2f_grid = STN_label(grid, flow_m2f).detach().cpu().numpy()[0, 0]
            #
            moving_img_nib = moving_img.detach().cpu().numpy()[0, 0]
            fixed_img_nib = fixed_img.detach().cpu().numpy()[0, 0]

            m2f_nib = m2f.detach().cpu().numpy()[0, 0]
            flow_m2f_nib = flow_m2f.detach().cpu().numpy()[0].astype(np.float32).transpose(1, 2, 3, 0)

            m2f_mask = m2f_mask.detach().cpu().numpy()
            multi_fixed_mask = multi_fixed_mask.detach().cpu().numpy()

            m2f_mask_nib = multi_class2mask(m2f_mask[0])
            multi_fixed_mask_nib = multi_class2mask(multi_fixed_mask[0])
            multi_moving_seg = mask2multi_class(moving_seg, 12).contiguous().cuda()
            multi_fixed_seg = mask2multi_class(fixed_seg, 12).contiguous().cuda()
            multi_moving_mask_nib = multi_class2mask(multi_moving_mask.detach().cpu().numpy()[0])
            multi_moving_seg_nib = multi_class2mask(multi_moving_seg.detach().cpu().numpy()[0])
            multi_fixed_seg_nib = multi_class2mask(multi_fixed_seg.detach().cpu().numpy()[0])



            # visualization
            save_dir = '/media/hao/DATA/ckpt/SAT/ACDC_ckpt_visual/'

            # input img
            moving_img_nib = nib.Nifti1Image(moving_img_nib, np.eye(4))
            fixed_img_nib = nib.Nifti1Image(fixed_img_nib, np.eye(4))

            m2f_nib = nib.Nifti1Image(m2f_nib, np.eye(4))
            flow_m2f_nib = nib.Nifti1Image(flow_m2f_nib, np.eye(4))
            m2f_mask_nib = nib.Nifti1Image(m2f_mask_nib, np.eye(4))

            multi_fixed_mask_nib = nib.Nifti1Image(multi_fixed_mask_nib, np.eye(4))
            multi_moving_mask_nib= nib.Nifti1Image(multi_moving_mask_nib, np.eye(4))
            multi_moving_seg_nib = nib.Nifti1Image(multi_moving_seg_nib, np.eye(4))
            multi_fixed_seg_nib = nib.Nifti1Image(multi_fixed_seg_nib, np.eye(4))
            m2f_grid_nib = nib.Nifti1Image(m2f_grid, np.eye(4))

            nib.save(m2f_nib, save_dir+"epoch_{}_{}_{}_moved_img.nii.gz".format( epoch_i,i+1, i+1))
            nib.save(flow_m2f_nib, save_dir+"epoch_{}_{}_{}_flow.nii.gz".format( epoch_i,i+1, i+1))
            nib.save(m2f_mask_nib, save_dir+"epoch_{}_{}_{}_predict.nii.gz".format( epoch_i,i+1, i+1))
            nib.save(multi_fixed_mask_nib, save_dir+"epoch_{}_{}_{}_fixed_label.nii.gz".format( epoch_i,i+1, i+1))
            nib.save(multi_moving_mask_nib,save_dir + "epoch_{}_{}_{}_moving_label.nii.gz".format(epoch_i,i + 1, i + 1))
            nib.save(moving_img_nib, save_dir + "epoch_{}_{}_{}_moving_img.nii.gz".format(epoch_i,i + 1, i + 1))
            nib.save(fixed_img_nib, save_dir + "epoch_{}_{}_{}_fixed_img.nii.gz".format(epoch_i,i + 1, i + 1))
            nib.save(multi_moving_seg_nib,save_dir + "epoch_{}_{}_{}_moving_pseudo.nii.gz".format(epoch_i,i + 1, i + 1))
            nib.save(multi_fixed_seg_nib,save_dir + "epoch_{}_{}_{}_fixed_pseudo.nii.gz".format(epoch_i,i + 1, i + 1))
            nib.save(m2f_grid_nib,save_dir + "epoch_{}_{}_{}_grid.nii.gz".format(epoch_i,i + 1, i + 1))

            dice_list = []
            for j in range(3):
                intersection = np.sum(m2f_mask[0, j] * multi_fixed_mask[0, j])
                smooth = 1e-4
                dice_3d = (2. * intersection) / (
                        np.sum(m2f_mask[0, j]) + np.sum(multi_fixed_mask[0, j]) + smooth)
                dice_list.append(dice_3d)
                # jac = jacobian_determinant(flow_m2f)
                # jac_det_list.append(torch.log((jac + 3).clamp_(0.000000001, 1000000000)).std().item())
            # print(pairs[i],dice_list)
            print(index_list, dice_list)

            final_dice.append(dice_list)

    print("over")
    dice_all = np.array(final_dice)
    dice_subject = np.mean(dice_all, axis=1)
    print('Per Subject DICE:', dice_subject)
    dice_all = np.mean(dice_all, axis=0)
    print('Per Class DICE:', dice_all)
    dice_all = np.mean(dice_all, axis=0)
    print('Final DICE:', dice_all)
    # jac_det_list = np.array(jac_det_list)
    # jac_all = np.mean(jac_det_list)
    # print('Final STD_J = {:4f}'.format(jac_all ))




if __name__ == "__main__":
    test()
