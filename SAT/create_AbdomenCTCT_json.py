import json
for i in range(30):

    image = "/media/hao/DATA/dataset/abdomen/abdomen/CT/imagesTr/AbdomenCTCT_{:0>4}_0000.nii.gz".format(i+1)
    mask = "/media/hao/DATA/dataset/abdomen/abdomen/CT/labelsTr/AbdomenCTCT_{:0>4}_0000.nii.gz".format(i+1)
    data = {"image": image,"mask": mask,
            "label": ["spleen","right kidney","left kidney","gallbladder", "esophagus","liver", "stomach", "aorta",
                      "inferior vena cava","portal vein and splenic vein","pancreas","right adrenal gland","left adrenal gland"], "modality": "ct"}


    json_str = json.dumps(data)

    with open("Abdomen_CT_CT_data.jsonl", "a+") as file:
        file.writelines([json_str,'\n'])
