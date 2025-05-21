import json
import os
import numpy as np
import scipy.io as sio
import json
import configparser


for i in range(1,101):
    dataPath = "/media/hao/DATA/dataset/ACDC/train/patient{:0>3d}".format(i)
    with open(os.path.join(dataPath, 'Info.cfg'), 'r') as f:
        data = f.read()
    if data[:9] != '[default]':
        with open(os.path.join(dataPath, 'Info.cfg'), 'w') as n:
            n.write('[default]\n')
            n.write(data)

    config = configparser.ConfigParser()
    config.read(os.path.join(dataPath, 'Info.cfg'))
    autodl = '/root/autodl-fs/ACDC/train/'
    ED = int(config.get('default', 'ED'))
    ES = int(config.get('default', 'ES'))
    dataA = os.path.join(autodl, os.path.basename(dataPath) + '_frame{:0>2d}.nii.gz'.format(ED))
    dataB = os.path.join(autodl, os.path.basename(dataPath) + '_frame{:0>2d}.nii.gz'.format(ES))
    label_dataA = os.path.join(autodl, os.path.basename(dataPath) + '_frame{:0>2d}_gt.nii.gz'.format(ED))
    label_dataB = os.path.join(autodl, os.path.basename(dataPath) + '_frame{:0>2d}_gt.nii.gz'.format(ES))



    data_a= {"image": dataA,"mask": label_dataA,
            "label": ["myocardium","left ventricle cavity", "right ventricle cavity"], "modality": "mri"}
    data_b = {"image": dataB, "mask": label_dataB,
              "label": ["myocardium", "left ventricle cavity", "right ventricle cavity"], "modality": "mri"}
    # 将字典转换为JSON字符串
    json_str_a = json.dumps(data_a)
    json_str_b = json.dumps(data_b)
    # 将JSON字符串写入文件
    with open("ACDC_MRI_data_AutoDL.jsonl", "a+") as file:
        file.writelines([json_str_a,'\n'])

    with open("ACDC_MRI_data_AutoDL.jsonl", "a+") as file:
        file.writelines([json_str_b, '\n'])