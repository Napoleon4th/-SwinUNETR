# SwinUNETR
医学人工智能 final project 作业
## 一、项目介绍
## 二、环境配置
该项目代码需要**MONAI**库辅助运行，**MONAI**（Medical Open Network for AI）是一个专为医疗影像AI设计的开源深度学习框架，基于PyTorch构建。可以使用conda安装，具体如下：
```shell
conda activate xxx（xxx为环境）
conda install -c conda-forge monai
```
安装后使用以下代码判断环境配置是否齐全：
```py
import os
import json
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch


print_config()
```

## 三、数据来源与预处理
**BraTS Challenge**（Brain Tumor Segmentation Challenge）是由MICCAI会议主办的国际公开挑战赛，专注于脑肿瘤的自动分割任务。本项目的训练与验证数据来自**BraTS 21 Challenge**，官方竞赛网站是https://www.synapse.org/#!Synapse:syn27046444/wiki/616992
不过由于比赛已结束，我个人并未在官网中找到数据下载入口。数据仍然可以从Kaggle下载，下载地址是https://www.kaggle.com/datasets/sadia1851/brain-tumor-detection-brast-2021 从kaggle下载的数据文件结构比较混乱，需要对文件进行预处理，请使用**Data_process.py**。

从kaggle上下载的文件夹解压后选择archive文件夹下BraTS2021_Training_Data文件夹中内容作为训练数据，共1251个病例样本，每个病例包含5个.nii文件（NIfTI格式，医学影像数据标准存储格式），后缀分别为：
```shell
_t1    用于显示正常解剖结构
_t1ce  在静脉注射钆对比剂后进行的T1加权扫描，主要作用于血脑屏障破坏区域
_t2    对水肿、炎症、囊肿死等液体含量增加的病变显示效果更佳
_flair FLAIR是一种特殊的T2加权技术，适用于脑白质病变、多发性硬化斑块、皮层下梗死、脑炎等疾病的诊断
_seg   提供的分割标签中，正常脑组织（NCR）的值为 1，水肿区（ED）的值为 2，肿瘤核心区（ET）的值为 4，其他区域的值为 0
```

仔细观察下载的数据，会发现需要的.nii文件被嵌套在子文件夹下，有两种嵌套的模式（这是由于kaggle下载的数据集结构紊乱，要自行处理）
```shell
1、嵌套结构：BraTS2021_00006/BraTS2021_00006_xxx.nii/00000116_final_seg.nii
2、扁平结构：BraTS2021_00006/BraTS2021_00006_xxx.nii
注意，嵌套结构和扁平结构可能会出现在同一个病例中，例如BraTS2021_00561文件夹下即有""BraTS2021_00561\BraTS2021_00561_seg.nii\BraTS2021_00561_seg_new.nii""也有"BraTS2021_00561\BraTS2021_00561_flair.nii"
```

我们希望最终数据结构都是\BraTS2021_00006\BraTS2021_00006_seg.nii.gz这样的格式，请在终端运行：
```shell
python Data_process.py --input "E:\数据集\Data\TrainingData"（存储数据的地址请按照自己情况修改，该地址下包含1251个病例文件夹）
```

## 四、实验

