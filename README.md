# IPMI25 - Medical Image Registration Meets Vision Foundation Model: Prototype Learning and Contour Awareness
>[Hao Xu](https://haoxu0507.github.io/), [Tengfei Xue](https://scholar.google.com/citations?user=VNalyUQAAAAJ&hl=en), [Jianan Fan](), [Dongnan Liu](https://scholar.google.com/citations?user=JZzb8XUAAAAJ&hl=zh-CN), [Yuqian Chen](https://scholar.google.com/citations?user=1RO71vMAAAAJ&hl=zh-CN), [Fan Zhang](https://scholar.google.com/citations?user=kTd978wAAAAJ&hl=zh-CN), [Carl-Fredrik Westin](https://scholar.google.com/citations?user=fUqBrO4AAAAJ&hl=zh-CN), [Ron Kikinis](https://scholar.google.com/citations?user=n01L0mEAAAAJ&hl=zh-CN), [Lauren J. O’Donnell](https://scholar.harvard.edu/laurenjodonnell/biocv), and [Weidong Cai](https://scholar.google.com/citations?user=N8qTc2AAAAAJ&hl=zh-CN) 
>
>*The 29th International Conference on Information Processing in Medical Imaging (IPMI) 2025 [[paper](https://arxiv.org/abs/2502.11440),[project](https://github.com/HaoXu0507/IPMI25-SAM-Assisted-Registration/),[code](https://github.com/HaoXu0507/IPMI25-SAM-Assisted-Registration/)]


![Poster](/Poster.jpg)

## Get Started

## Datasets Prepare
You can prepare datasets by yourself or follow the following steps.
* Download [the Abdomen CT Dataset](https://learn2reg.grand-challenge.org/Datasets/).
* Download [the ACDC MRI Dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).

## Pre-trained SAM Model Prepare 
* Download the pretrained [SAT-Nano/Pro model](https://github.com/zhaoziheng/SAT).

## SAM Mask Generation
1. Make the JSON file by running [create_Abdomen_CTCT_json.py](/SAT/create_Abdomen_CTCT_json.py)
   
2. Generate Masks according to the text prompt by running [inference_Abdomen.py](/SAT/inference_Abdomen.py)

## Training
``python /Reg_Model/train.py ``

## Testing
``python /Reg_Model/test.py ``

## Visualization
![Visualization](/Visualization.png)

## Citation

```
@inproceedings{Xu_2025_IPMI,
  title={MultiCo3D: Multi-Label Voxel Contrast for One-Shot Incremental Segmentation of 3D Neuroimages},
  author={{Hao Xu and Tengfei Xue and Dongnan Liu and Yuqian Chen and Fan Zhang and Carl-Fredrik Westin and Ron Kikinis and Lauren J. O'Donnell and Weidong Cai},
  booktitle={The 29th International Conference on Information Processing in Medical Imaging (IPMI) 2025},
  year={2025},
}
```


