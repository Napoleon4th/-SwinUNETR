# SwinUNETR
医学人工智能 final project 作业
## 一、项目介绍
Swin UNETR（Swin UNEt TRansformers）是一种先进的3D医学图像分割模型，于2022年MICCAI会议上提出。它将Swin Transformer作为encoder，用于高效捕捉全局长距离依赖和多尺度特征，同时结合CNN-based decoder通过跳跃连接恢复局部细节。该模型专为多模态MRI脑瘤分割设计，在BraTS 2021挑战赛验证阶段排名前列，显著优于传统CNN方法如nnU-Net。

本项目基于MONAI框架复现了Swin UNETR在BraTS 2021数据集上的脑瘤分割任务。代码参考MONAI官方教程和研究贡献仓库，实现完整训练流程，包括数据加载、增强、模型训练、验证和结果可视化。为适应单机环境（8GB显卡），对官方配置进行了优化。项目成功实现了从零训练、模型保存和损失/Dice曲线绘制，展示了Transformer在3D医学图像分割中的强大潜力。
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
### （1）数据来源
**BraTS Challenge**（Brain Tumor Segmentation Challenge）是由MICCAI会议主办的国际公开挑战赛，专注于脑肿瘤的自动分割任务。本项目的训练与验证数据来自**BraTS 21 Challenge**，官方竞赛网站是https://www.synapse.org/#!Synapse:syn27046444/wiki/616992
不过由于比赛已结束，我个人并未在官网中找到数据下载入口。数据仍然可以从Kaggle下载，下载地址是https://www.kaggle.com/datasets/sadia1851/brain-tumor-detection-brast-2021 从kaggle下载的数据文件结构比较混乱，需要对文件进行预处理，请使用**Data_process.py**。

从kaggle上下载的文件夹解压后选择archive文件夹下BraTS2021_Training_Data文件夹中内容作为训练数据，共1251个病例样本，每个病例包含5个.nii文件（NIfTI格式，医学影像数据标准存储格式），后缀分别为：
```shell
_t1    用于显示正常解剖结构
_t1ce  在静脉注射钆对比剂后进行的T1加权扫描，主要作用于血脑屏障破坏区域
_t2    对水肿、炎症、囊肿死等液体含量增加的病变显示效果更佳
_flair FLAIR是一种特殊的T2加权技术，适用于脑白质病变、多发性硬化斑块、皮层下梗死、脑炎等疾病的诊断
_seg   提供的分割标签中，肿瘤的坏死部分（NCR）的值为 1，脑周围水肿/受侵组织（ED）的值为 2，强化肿瘤（ET）的值为 4，其他区域的值为 0
```
进行评估的目标包括“强化肿瘤”（ET）、“肿瘤核心”（TC）和“整个肿瘤”（WT）。ET是在与T1相比时，在T1CE中显示出高信号强度的区域；TC包含ET以及肿瘤的坏死（NCR）部分；WT描述的是疾病的全部范围，包括TC和肿瘤周围的水肿/受侵组织（ED）。
### （2）数据预处理
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
请将**package.json**文件置于数据集文件夹同一个文件夹中，例如"E:\数据集\Data\TrainingData"为数据集，"E:\数据集\Data\package.json"为json文件位置，用于交叉验证。

## 四、实验
**运行main.py进行实验**
### （1）实验流程
#### 数据加载：
使用 MONAI 的LoadImaged加载多模态图像和标签，将标签转换为 3 通道（对应 ET、TC、WT）。训练时进行随机裁剪、随机翻转、强度归一化、随机强度缩放和偏移等数据增强。验证时仅进行归一化。
#### 模型训练：
每epoch遍历训练 DataLoader，进行前向传播、计算 DiceLoss、反向传播并更新参数。使用滑动窗口推理（sliding_window_inference，SwinTransformer特点）在验证阶段进行全体积预测。

**Dice Score是评估指标，用于衡量两个样本集合相似度的统计量，即计算预测结果与真实结果之间像素点的重叠程度。**
#### 验证与模型保存：
每 2 个 epoch进行一次验证，计算三个子区域的 Mean Dice分数。当验证集平均 Dice 分数最高时保存最佳模型检查点（checkpoint）。
#### 训练结束：
输出最佳平均 Dice 分数。绘制训练损失曲线和验证 Dice 曲线（整体平均值和三个子区域共四张 Dice 图像）。

### （2）主要超参数

| 参数                  | 值                  | 说明                                      |
|-----------------------|---------------------|-------------------------------------------|
| roi_size              | (96, 96, 96)       | 输入 patch 大小（设置比官方小，节省显存）      |
| batch_size            | 2                   | 训练批次大小                              |
| sw_batch_size         | 2                   | 滑动窗口推理批次大小                      |
| infer_overlap         | 0.5                 | 滑动窗口重叠率                            |
| max_epochs            | 30                  | 最大训练轮数，一个epoch训练时间很长                              |
| val_every             | 2                   | 每多少 epoch 进行一次验证                  |
| feature_size          | 24                  | Swin Transformer 特征维度（默认 48，此处减小以适应显存） |
| use_checkpoint        | True                | 启用梯度检查点，用于降低显存占用           |
| learning_rate         | 1e-4                | AdamW 初始学习率                          |
| optimizer             | AdamW               | 优化器（weight_decay=1e-5）                |
| scheduler             | CosineAnnealingLR   | 学习率调度器                              |

### （3）预期结果
损失会逐步下降，验证 Dice 分数逐步上升。通常在前几十个 epoch 内快速提升，后期趋于平稳。

**注意：由于本复现使用了较小的 feature_size=24 和 roi_size=96x96x96（官方常为 48 和 128），以及单块GPU，最终 Dice 分数会略低于官方 Swin UNETR 在 BraTS 2021 验证阶段的 ~0.913。**
