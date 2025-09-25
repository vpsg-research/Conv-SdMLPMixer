<div align="center">


<h1>【Information Fusion 2025】Conv-SdMLPMixer: A Hybrid Medical Image Classification Network Based on Multi-branch CNN and Multi-scale multi-dimensional MLP</h1>

</div>  <!-- ✅ 这一行必须加，结束居中区域 -->

## ⭐ Abstract


Addressing the common issues of high noise and relatively small lesion areas in medical image datasets, this paper proposes a novel hybrid network model named Conv-SdMLPMixer. This model combines the strengths of Convolutional Neural Networks (CNNs) and Multilayer Perceptrons (MLPs). Specifically, the paper first introduces a multi-path inverted residual bottleneck CNN (MIRB-CNN) structure designed to enhance the network's ability to focus on lesion area details. Secondly, to overcome the limited receptive field of CNNs, the paper innovatively incorporates a multi-scale and multi-dimensional feature fusion MLP (SdMLP) module. This module employs three key operations: overlapping multi-scale patches (OMSP), spatial-channel MLP (ScMLP), and feature aggregation nodes, to enable the network to capture information from three different dimensions: rows, columns, and channels on overlapping feature maps. Experimental results on the BUSI, COVID19-CT, Small-ISIC2018, and Chest-Xray datasets confirm that Conv-SdMLPMixer outperforms existing technologies. Particularly on the BUSI dataset, Conv-SdMLPMixer achieved an accuracy of 93.33%, an F1 score of 93.65%, a precision rate of 94.78%, and a recall rate of 92.66%, which fully demonstrates its outstanding performance in medical image classification tasks. Through in-depth comparison and analysis of CNN and MLP architectures, this paper has verified the advantages of the hybrid network in medical image classification tasks, providing strong technical support for precise medical diagnosis.

## 📻 Overview

<div align="center">
    <img width="1000" alt="image" src="image\1.png">
</div>

<div align="center">
The overall architecture of Conv-SdMLPMixer, which consists of MIRB-CNN block and SdMLP block.
</div>

## 📆 Release Plan

- [x] Project page released
- [x] Dataset preparation instructions released
- [x] Model code released
- [x] Training and evaluation scripts released

## 📁 Dataset Preparation
### 1\. Download the Datasets

  * **BUSI**: [Download Link](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
  * **COVID19-CT**: [Download Link](https://github.com/emi-dm/COVID-CT-Dataset)
  * **Small-ISIC2018**: [Download Link](https://github.com/RuiZhang97/ISNet)
  * **Chest-Xray**: [Download Link](https://github.com/RuiZhang97/ISNet)


### 2\. Organize the Directory Structure
After downloading and extracting the files, place them in the `datasets` folder located in the project's root directory. Your project should adhere to the following structure:

```
Conv-SdMLPMixer/
├── datasets/
│   ├── BUSI/
│   │   ├── train/
│   │   │   ├── begign/
│   │   │   │   └──.png
│   │   │   └── ...
│   │   └── tset/
│   │       ├── begign/
│   │       │   └──.png
│   │       └── ...
│   │
│   ├── COVID19-CT/
│   │   ├── train/
│   │   │   ├── CT_COVID/
│   │   │   │   └──.png
│   │   │   └── ...
│   │   └── tset/
│   │       ├── CT_NonCOVID/
│   │       │   └──.png
│   │       └── ...
│   │
│   ├── Small-ISIC2018/
│   │   ├── train/
│   │   │   ├── begign/
│   │   │   │   └──.jpg
│   │   │   └── ...
│   │   └── tset/
│   │       ├── begign/
│   │       │   └──.jpg
│   │       └── ...
│   │
│   └── Chest-Xray/
│       ├── train/
│       │   ├── 0/
│       │   │   └──.jpeg
│       │   └── ...
│       └── test/
│           ├── 0/
│           │   └──.jpeg
│           └── ...
│
├── train.py
├── test.py
└── model.py
```


### 💡 How to Use

### 1\. Training

Run the following command to start training the model. The script will automatically load the data from the `datasets` directory.

```bash
python train.py
```

### 2\. Testing

Use the `test.py` script to evaluate the model's performance. Please ensure you have completed the training process.

```bash
python test.py
```

## Citation
Please cite our paper if the code is used in your research:
```
@article{REN2025102937,
title = {Conv-SdMLPMixer: A hybrid medical image classification network based on multi-branch CNN and multi-scale multi-dimensional MLP},
journal = {Information Fusion},
volume = {118},
pages = {102937},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.102937},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525000107},
author = {Zitong Ren and Shiwei Liu and Liejun Wang and Zhiqing Guo}.
}
```
