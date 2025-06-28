# ----- config.py -----
import torch
import os

# 数据与路径配置
# dataset/labels csv 文件和 photo 文件夹路径
dataset_dir = "dataset"
csv_path = os.path.join(dataset_dir, "labels_synthetic_all.csv")
img_folder = os.path.join(dataset_dir, "photo")
# img_folder = "dataset/photo"
# 图像预处理与热力图参数
# 原图约 4000×3000，缩到 1024×768 可将量化误差控制在 ≈3–4 px
IMG_SIZE = (768, 1024)   # (H, W) 维持 4:3 比例
SIGMA = 4                # 高斯核 σ，缩放后角点直径≈10‑15 px，σ≈6 最合适
NUM_KPTS = 4             # 关键点数量

# 训练超参
# 32 GB 显存 (M2 Max) 下，batch=4 对 1024×768 U‑Net 合理
BATCH_SIZE = 4
EPOCHS = 30              # 适当延长训练轮次
LR = 3e-4                # 较大分辨率下稍降学习率

# 优化器与调度器配置
# 使用 AdamW 并添加 weight decay 以增强正则化
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-5

# 学习率调度器，可选 "CosineAnnealingLR" 或 "ReduceLROnPlateau"
SCHEDULER = "CosineAnnealingLR"
# 如果使用 CosineAnnealingLR，可进一步配置如下：
# SCHEDULER_PARAMS = {"T_max": EPOCHS, "eta_min": 1e-6}
# 如果使用 ReduceLROnPlateau，可配置：
# SCHEDULER = "ReduceLROnPlateau"
# SCHEDULER_PARAMS = {"mode": "min", "factor": 0.5, "patience": 3}

# 设备
DEVICE = torch.device("mps")