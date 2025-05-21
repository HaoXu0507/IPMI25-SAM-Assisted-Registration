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
from scipy.interpolate import Rb


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





def mk_grid_img(grid_step=20, line_thickness=1, grid_sz=[160, 192, 160]):
    grid_img = np.zeros(grid_sz)
    for j in range(0 + grid_step, grid_img.shape[0], grid_step):
        grid_img[j:j + line_thickness, :, :] = 1
    for i in range(0 + grid_step, grid_img.shape[1], grid_step):
        grid_img[:, i:i + line_thickness, :] = 1
    # for i in range(0, grid_img.shape[2], grid_step):
    #     grid_img[:, :, i+line_thickness-1] = 1

    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img



def mask2multi_class(mask, mask_max=15):
    # mask_max = mask.max()
    N, H, W, D = mask.shape
    multi_mask = torch.zeros((N, mask_max, H, W, D))
    for i in range(mask_max):
        multi_mask[:, i] = mask[0] == i + 1
    return multi_mask


def multi_class2mask(mask):
    # mask_max = mask.max()
    c, H, W, D = mask.shape
    out = np.zeros((H, W, D))
    for i in range(c):
        out = np.where(mask[i] == 1, np.ones((H, W, D)) * (i + 1), out)
    out = out.astype(np.float)
    return out




def InfoNCE_Loss(embedding, sample_list, temperature=1):
    # size = sample_list.size
    # b,n,c = size[0],size[1],size[2]
    b, n, c = sample_list.shape

    loss_con = torch.tensor([0.0]).cuda()
    for batch in range(b):
        batch_sample = sample_list[batch]
        anchor = batch_sample[0]
        positive = batch_sample[1]
        # hard_negative = batch_sample[2:7]
        # easy_negative = batch_sample[7:12]
        negative = batch_sample[2:]

        embedding_distance_set = []

        embedding_distance_set.append(
            torch.nn.functional.pairwise_distance(embedding[0, :, anchor[0], anchor[1], anchor[2]],
                                                  embedding[0, :, positive[0], positive[1], positive[2]]))
        for i in range(len(negative)):
            embedding_distance_set.append(
                torch.nn.functional.pairwise_distance(embedding[0, :, anchor[0], anchor[1], anchor[2]],
                                                      embedding[0, :, negative[i, 0], negative[i, 1], negative[i, 2]]))
        numerator = torch.exp(embedding_distance_set[0] / temperature)
        denominator = torch.tensor(0.0).cuda()
        for j in range(len(negative)):
            denominator += torch.exp(embedding_distance_set[j + 1] / temperature)
        loss = -torch.log(numerator / denominator)
        loss_con += loss
    loss_con = loss_con / b
    # loss_con = torch.div(loss_con, torch.tensor(b).cuda())
    return loss_con


def sample_and_calculate_loss(seg, embedding):
    sample_set = []
    for class_index in range(12):
        sample_class_set = []
        # anchor is in the inner circle
        # positive is in the inner boundery
        # negative is in the outer boundery
        this_lab = seg[0, class_index].cpu().numpy()
        # if (this_lab == np.zeros((160, 160, 160))).all():
        h, w, d = this_lab.shape
        if (this_lab == np.zeros_like(this_lab)).all():

            print(class_index, 'pass')

            continue
        else:
            kernel = np.ones((3, 3, 3), np.uint8)

            inner_lab = binary_erosion(this_lab, structure=kernel)
            outer_lab = binary_dilation(this_lab, structure=kernel)

            if not np.all(inner_lab == 0):

                # anchor and simple positive 5+1=6 coordiantes
                while len(sample_class_set) < 1:
                    coordinate = np.random.randint(low=0, high=h, size=[2]).tolist()
                    coordinate.append(np.random.randint(h))
                    # print(inner_lab[coordinate[0],coordinate[1]])
                    if inner_lab[coordinate[0], coordinate[1], coordinate[2]] == 1:
                        sample_class_set.append(list(coordinate))

                # hard positive 5 coordiantes
                while len(sample_class_set) < 2:
                    coordinate = np.random.randint(0, h, size=[2]).tolist()
                    coordinate.append(np.random.randint(h))
                    if inner_lab[coordinate[0], coordinate[1], coordinate[2]] == 0 and this_lab[
                        coordinate[0], coordinate[1], coordinate[2]] == 1:
                        sample_class_set.append(list(coordinate))

                # hard negative 5 coordiantes
                while len(sample_class_set) < 7:
                    coordinate = np.random.randint(0, h, size=[2]).tolist()
                    coordinate.append(np.random.randint(h))
                    if outer_lab[coordinate[0], coordinate[1], coordinate[2]] == 1 and this_lab[
                        coordinate[0], coordinate[1], coordinate[2]] == 0:
                        sample_class_set.append(list(coordinate))

                # out of circle 5 coordiantes
                while len(sample_class_set) < 12:
                    coordinate = np.random.randint(0, h, size=[2]).tolist()
                    coordinate.append(np.random.randint(h))
                    if outer_lab[coordinate[0], coordinate[1], coordinate[2]] == 0:
                        sample_class_set.append(list(coordinate))

        if len(sample_class_set) != 0:
            sample_set.append(sample_class_set)
            # print(class_index)

    sample_set = np.array(sample_set)
    loss_con_moving_seg = InfoNCE_Loss(embedding, (sample_set / 4).astype(int))
    return loss_con_moving_seg



def train():
    # for Abdomen CT
    json_dir = '/home/hao/PycharmProjects/SAT-main/ACDC/TransMatch_TMI-main/ACDC_image_seg_data.jsonl'
    data = prepare_data_ACDC(json_dir=json_dir, split='train')
    vol_size = [160, 160, 160]

    # for ACDC MRI
    # json_dir = '/home/hao/PycharmProjects/SAT-main/TransMatch_TMI-main/segmentation_result_Abdomen_CT_CT_data.jsonl'
    # data = prepare_data(json_dir = json_dir,split='train')
    # vol_size = [128, 128, 32]

    device = torch.device('cuda')

    from Models.Conv_Transformer_Reg import, CNN_Attention_Net
    net = CNN_Attention_Net().to(device)
    import TransMorph_models.TransMorph_bspl as TransMorph_bspl

    epoch = 500

    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

    net.train()
    STN.train()

    scaler = GradScaler()
    optimizer = Adam(net.parameters(), lr=0.0001)
    optimizer.load_state_dict(ckpt['optimizer'])

    sim_loss_fn = losses.ncc_loss
    # sim_loss_fn = losses.mse_loss
    # sim_loss_fn = losses.MIND_loss()
    # sim_loss_fn = losses.MutualInformation()

    grad_loss_fn = losses.gradient_loss
    dice_loss_fn = losses.Dice()

    for epoch_i in range(0 + 1, epoch + 1):

        N, _, H, W, D = data['images'].shape
        pairs = data['pairs']
        pair_max = len(pairs)
        index_list = np.array(range(pair_max))
        random.shuffle(index_list)
        idx = 0
        epoch_dice = 0

        # train
        for i in index_list:
            idx += 1

            moving_img = data['images'][pairs[i][0]]
            moving_seg = data['segmentations'][pairs[i][0]]

            fixed_img = data['images'][pairs[i][1]]
            fixed_seg = data['segmentations'][pairs[i][1]]

            moving_img = torch.from_numpy(moving_img).cuda().unsqueeze(0)
            moving_seg = torch.from_numpy(moving_seg).cuda()
            fixed_img = torch.from_numpy(fixed_img).cuda().unsqueeze(0)
            fixed_seg = torch.from_numpy(fixed_seg).cuda()

            multi_moving_seg = mask2multi_class(moving_seg, 3).contiguous().cuda()
            multi_fixed_seg = mask2multi_class(fixed_seg, 3).contiguous().cuda()

            moving_img = moving_img.float()
            fixed_img = fixed_img.float()
            with autocast():
                out = net(moving_img.float(), fixed_img.float(), moving_seg.float().unsqueeze(0),
                          fixed_seg.float().unsqueeze(0), )
                fieid4, fieid4_up, fieid3, fieid3_up, fieid2, fieid2_up, fieid1, fieid0, field = out['field']

                flow_m2f = field

                m2f = STN(moving_img, flow_m2f)
                m2f_label = STN(multi_moving_seg, flow_m2f)

                # we change Chamfer loss to Hausdorff loss
                loss_hausdorff = monai.losses.HausdorffDTLoss(include_background=False, to_onehot_y=False,
                                                              sigmoid=False, alpha=2.0)(m2f_label, multi_fixed_seg)

                # H * W * D = 128 * 128 * 32
                # 1/4 size* 96, 1/8 size*192, 1/16size*384, 1/32size*768
                I_f_feature1, I_f_feature2, I_f_feature3, I_f_feature4 = out['fixed_img_feature']
                I_m_feature1, I_m_feature2, I_m_feature3, I_m_feature4 = out['moving_img_feature']
                S_f_feature1, S_f_feature2, S_f_feature3, S_f_feature4 = out['fixed_mask_feature']
                S_m_feature1, S_m_feature2, S_m_feature3, S_m_feature4 = out['moving_mask_feature']

                multi_moving_seg4 = F.interpolate(multi_moving_seg, scale_factor=0.25, mode='nearest').int()
                multi_fixed_seg4 = F.interpolate(multi_fixed_seg, scale_factor=0.25, mode='nearest').int()

                loss_contrast_im = sample_and_calculate_loss(multi_moving_seg4, I_m_feature4)
                loss_contrast_if = sample_and_calculate_loss(multi_fixed_seg4, I_f_feature4)

                loss_contrastive = 0.01 * (loss_contrast_im + loss_contrast_if) / 2

                sim_loss = sim_loss_fn(m2f, fixed_img)
                grad_loss = grad_loss_fn(flow_m2f)

                dice_loss = dice_loss_fn(m2f_label, multi_fixed_seg)

                loss = 1 * dice_loss + 1 * grad_loss + 4 * sim_loss + 0.1 * loss_hausdorff + 1 * loss_contrastive
                
                print(
                    'Epoch {} Iter {}/{} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Img Seg: {:.6f}, Hausdorff: {:.6f}'.format(
                        epoch_i, idx, pair_max, loss.item(), sim_loss.item(), grad_loss.item(), dice_loss.item(),
                        loss_hausdorff.item()))

                # Backwards and optimize
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            epoch_dice += dice_loss.detach().cpu().numpy()
        epoch_dice = epoch_dice / pair_max
        print("epoch dice = {}".format(epoch_dice))
        os.system("nvidia-smi")
        if epoch_i %10 == 0 or epoch_i==1 or epoch_i>epoch-10:

            torch.save({
                'state_dict': net,
                'optimizer': optimizer.state_dict(),
            },
                '/ckpt/epoch_{}.pth.tar'.format(epoch_i)
            )



if __name__ == "__main__":
    train()
