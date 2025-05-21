import os

import torch
import torch.nn as nn

from .text_tower import Text_Tower
from .SAT import SAT

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def load_text_encoder(args, device, gpu_id):
    model = Text_Tower(args.tokenizer_path, 768)
    
    model = model.to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    #
    checkpoint = torch.load(args.text_encoder_checkpoint, map_location=device)

    model_state = model.state_dict()
    ckpt_model_state = checkpoint['model_state_dict']

    # model.load_state_dict(ckpt_model_state, strict=False)

    for k, v in model_state.items():
        for k2, v2 in ckpt_model_state.items():
            if k == k2[7:]:
                model_state[k] = ckpt_model_state[k2]

    model.load_state_dict(model_state, strict=False)

    # model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    # if int(os.environ["RANK"]) == 0:
    print(f"** Model ** Load pretrained text encoder from {args.text_encoder_checkpoint}.")
        
    return model

def build_segmentation_model(args, device, gpu_id):
    model = SAT(args.vision_backbone, args.patch_size)
    
    model = model.to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
            
    return model
