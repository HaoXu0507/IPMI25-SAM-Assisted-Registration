import os
import datetime
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from einops import reduce, rearrange, repeat

from dataset.inference_dataset import Inference_Dataset, inference_collate_fn
from model.tokenizer import MyTokenizer
from model.build_model import load_text_encoder, build_segmentation_model


def compute_gaussian(tile_size, sigma_scale: float = 1. / 8, value_scaling_factor: float = 10, dtype=np.float16):
    # build an gaussian filter with the patch size
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def inference(model, save_dir, tokenizer, text_encoder, device, testloader, gaussian_filter):
    # the left/right of data is reverse comparing with the sat model
    # inference
    model.eval()
    text_encoder.eval()
    mask_label = ['background',
                   "right ventricle cavity","myocardium","left ventricle cavity", ]

    text_label =["myocardium","left ventricle cavity", "right ventricle cavity"]

    with torch.no_grad():


        testloader = tqdm(testloader, disable=False)

        # gaussian kernel to accumulate predcition
        windows = compute_gaussian((288, 288, 96)) if gaussian_filter else np.ones((288, 288, 96))
        dice = []
        for batch in testloader:  # one batch for each sample
            # data loading
            batched_patch = batch['batched_patch']
            batched_y1y2x1x2z1z2 = batch['batched_y1y2x1x2z1z2']
            split_prompt = batch['split_prompt']
            split_n1n2 = batch['split_n1n2']
            labels = batch['labels']
            image = batch['image']
            mask = batch['mask']
            mask_path = batch['mask_path']
            image_path = batch['image_path']

            _, h, w, d = image.shape

            n = split_n1n2[-1][-1]
            prediction = np.zeros((n, h, w, d))
            accumulation = np.zeros((n, h, w, d))

            # for each batch of queries
            for prompts, n1n2 in zip(split_prompt, split_n1n2):
                n1, n2 = n1n2
                input_text = tokenizer.tokenize(prompts)  # (max_queries, max_l)
                input_text['input_ids'] = input_text['input_ids'].to(device=device)
                input_text['attention_mask'] = input_text['attention_mask'].to(device=device)
                queries, _, _ = text_encoder(text1=input_text, text2=None)  # (max_queries, d)

                # for each batch of patches
                for patches, y1y2_x1x2_z1z2_ls in zip(batched_patch, batched_y1y2x1x2z1z2):  # [b, c, h, w, d]
                    batched_queries = repeat(queries, 'n d -> b n d', b=patches.shape[0])  # [b, n, d]
                    patches = patches.to(device=device)

                    prediction_patch = model(queries=batched_queries, image_input=patches)
                    prediction_patch = torch.sigmoid(prediction_patch)
                    prediction_patch = prediction_patch.detach().cpu().numpy()  # bnhwd

                    # fill in
                    for b in range(len(y1y2_x1x2_z1z2_ls)):
                        y1, y2, x1, x2, z1, z2 = y1y2_x1x2_z1z2_ls[b]

                        # accumulation
                        prediction[n1:n2, y1:y2, x1:x2, z1:z2] += prediction_patch[b, :n2 - n1, :y2 - y1, :x2 - x1,
                                                                  :z2 - z1] * windows[:y2 - y1, :x2 - x1, :z2 - z1]
                        accumulation[n1:n2, y1:y2, x1:x2, z1:z2] += windows[:y2 - y1, :x2 - x1, :z2 - z1]

            # avg
            prediction = prediction / accumulation
            prediction = np.where(prediction > 0.5, 1.0, 0.0)

            # save prediction
            save_name = image_path.split('/')[-1].split('.')[0]  # xxx/xxx.nii.gz --> xxx/xxx
            np_images = image.numpy()[0, :, :, :]
            Path(save_dir + '/' + save_name).mkdir(exist_ok=True, parents=True)
            results = np.zeros((h, w, d))  # merge on one channel
            # calculate dice
            dice_list = []
            for j, label in enumerate(labels):
                if label not in ["adrenal gland", "colon", "duodenum", "small bowel"]:
                    seg_result = prediction[j, :, :, :]
                    # mask_label_index = j+1
                    for m_i, m_l in enumerate(mask_label):
                        if m_l == label:
                            mask_label_index = m_i
                            break
                else:
                    continue
                mask_j = np.where(mask.numpy()[0, :, :, :] == mask_label_index, 1, 0)

                intersection = np.sum(mask_j * seg_result)
                smooth = 1e-4
                dice_3d = (2. * intersection + smooth) / (np.sum(mask_j) + np.sum(seg_result) + smooth)
                dice_list.append(dice_3d)
            print(dice_list)
            dice.append(dice_list)

            # save for visualization
            for j, label in enumerate(labels):
                results += prediction[j, :, :, :] * (j+1)   # 0 --> 1 (skip background)
                pred_obj = nib.nifti2.Nifti1Image(prediction[j, :, :, :], np.eye(4))
                nib.save(pred_obj, f'{save_dir}/{save_name}/{label}.nii.gz')

            pred_obj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(pred_obj, f'{save_dir}/{save_name}/prediction.nii.gz')

            # save image
            imgobj = nib.nifti2.Nifti1Image(np_images, np.eye(4))
            nib.save(imgobj, f'{save_dir}/{save_name}/image.nii.gz')

            # save mask
            mask_obj = nib.nifti2.Nifti1Image(mask.numpy()[0, :, :, :], np.eye(4))
            nib.save(mask_obj, f'{save_dir}/{save_name}/mask.nii.gz')


            pred_obj = nib.nifti2.Nifti1Image(results, np.eye(4))
            nib.save(pred_obj, f'{save_dir}/{save_name}/prediction.nii.gz')

            # save image
            imgobj = nib.nifti2.Nifti1Image(np_images, np.eye(4))
            nib.save(imgobj, f'{save_dir}/{save_name}/image.nii.gz')

            # save mask
            mask_obj = nib.nifti2.Nifti1Image(mask.numpy()[0, :, :, :], np.eye(4))
            nib.save(mask_obj, f'{save_dir}/{save_name}/mask.nii.gz')

        print("over")
        dice_all = np.array(dice)
        dice_all = np.mean(dice_all, axis=0)
        print('Per Class DICE:', dice_all)
        dice_all = np.mean(dice_all, axis=0)
        print('Final DICE:', dice_all)


def main(args):
    # set gpu
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = device = torch.device("cuda")
    gpu_id = 0

    # dataset and loader
    testset = Inference_Dataset(args.data_jsonl, args.patch_size, args.max_queries, args.batchsize)
    # sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=None, batch_size=1, collate_fn=inference_collate_fn, shuffle=False)
    # sampler.set_epoch(0)

    # set segmentation model
    model = build_segmentation_model(args, device, gpu_id)

    # load checkpoint of segmentation model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_state = model.state_dict()
    ckpt_model_state = checkpoint['model_state_dict']

    # model.load_state_dict(ckpt_model_state, strict=False)

    for k, v in model_state.items():
        for k2, v2 in ckpt_model_state.items():
            if k == k2[7:]:
                model_state[k] = ckpt_model_state[k2]

    model.load_state_dict(model_state, strict=False)

    # if int(os.environ["RANK"]) == 0:
    print(f"** Model ** Load segmentation model from {args.checkpoint}.")

    # load text encoder
    text_encoder = load_text_encoder(args, device, gpu_id)

    # set tokenizer
    tokenizer = MyTokenizer(args.tokenizer_path)

    # choose how to evaluate the checkpoint
    inference(model, args.save_dir, tokenizer, text_encoder, device, testloader, args.gaussian_filter)


if __name__ == '__main__':
    '''
    --data_jsonl
    /home/hao/PycharmProjects/SAT-main/ACDC/ACDC_MRI_data.jsonl
    --checkpoint
    /media/hao/DATA/download/SAT_Nano2.pth
    --text_encoder_checkpoint
    /media/hao/DATA/download/text_encoder.pth
     --save_dir
    /media/hao/DATA/dataset/SAT/Nano/ACDC/
    '''
    def str2bool(v):
        return v.lower() in ('true')


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/media/hao/DATA/ckpt/SAT_Nano/ACDC/',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path of SAT",
    )
    parser.add_argument(
        "--text_encoder_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path of text encoder",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_jsonl",
        type=str,
        help="Path to jsonl file",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs='+',
        default=[288, 288, 96],
        help='Size of input patch',
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default='UNET',
    )
    parser.add_argument(
        "--gaussian_filter",
        type=str2bool,
        default='False',
    )
    args = parser.parse_args()

    main(args)