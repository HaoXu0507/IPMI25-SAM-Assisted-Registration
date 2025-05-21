# IPMI25 - Medical Image Registration Meets Vision Foundation Model: Prototype Learning and Contour Awareness
>[Hao Xu](https://haoxu0507.github.io/), [Tengfei Xue](https://scholar.google.com/citations?user=VNalyUQAAAAJ&hl=en), [Jianan Fan](), [Dongnan Liu](https://www.researchgate.net/profile/Dongnan-Liu), [Yuqian Chen](https://scholar.google.com/citations?user=1RO71vMAAAAJ&hl=zh-CN), [Fan Zhang](https://scholar.harvard.edu/fanzhang), [Carl-Fredrik Westin](https://brighamandwomens.theopenscholar.com/lmi/people/carl-fredrik-westin), [Ron Kikinis](https://brighamandwomens.theopenscholar.com/lmi/people/ron-kikinis-md), [Lauren J. O’Donnell](https://scholar.harvard.edu/laurenjodonnell/biocv), and [Weidong Cai](https://weidong-tom-cai.github.io/) 
>
>*The 29th International Conference on Information Processing in Medical Imaging (IPMI) 2025 [[paper](https://arxiv.org/abs/2502.11440),[project],[code](https://github.com/HaoXu0507/IPMI25-SAM-Assisted-Registration/)]


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


