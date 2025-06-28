# train_yolo_pose.py
# ----------------------------------------------------
# 用 Ultralytics YOLOv8-Pose 训练 4-点绿色角点模型
# 数据目录结构:
# dataset/
# ├─ images/       ← 所有 JPEG / PNG
# ├─ labels/       ← 与 images 同名 .txt (CSV 已转)
# └─ data.yaml     ← 你粘贴的配置 (见下)
# ----------------------------------------------------

from ultralytics import YOLO
# from pathlib import Path

model = YOLO("yolov8l-pose.pt")

# 2️⃣ 训练设置
EPOCHS      = 120             # 训练轮数（总共迭代整个数据集的次数）
IMGSZ       = 960             # 输入图像尺寸，正方形边长，影响模型感受野和计算量
BATCH       = 8               # 每次迭代加载的图像数量，影响显存占用和梯度稳定性
LR0         = 2e-5            # 初始学习率，控制参数更新步长
WEIGHT_DEC  = 5e-6            # 权重衰减系数，用于 L2 正则化，抑制过拟合
DATA_YAML   = "/Users/charon/Desktop/大学/Python/备用矫正方案/yolo_dataset/data.yaml"  # 数据集配置文件路径，包含 train/val 列表和类别定义

# 3️⃣ 开始训练
if __name__ == "__main__":
    model.train(
        mosaic=0.0,  # 关掉 Mosaic
        mixup=0.0,
        save_period=5,
        data=DATA_YAML,         # 数据集的 data.yaml 文件路径
        epochs=EPOCHS,          # 总训练轮数
        imgsz=IMGSZ,            # 输入图像边长（像素）
        batch=BATCH,            # 每个批次的图像数量
        lr0=LR0,                # 初始学习率
        optimizer="AdamW",      # 优化器类型，AdamW 支持权重衰减
        weight_decay=WEIGHT_DEC,# 优化器权重衰减参数
        lrf=1e-6,               # 学习率最终衰减系数（lr_final = lr0 * lrf）
        warmup_epochs=3,
        device="mps",           # 训练设备，mps on Apple Silicon
        kobj=2.5,               # 关键点置信度损失权重
        box=0.0,                # 边界框损失权重，设为0表示不更新box分支
        cls=0.0,                # 分类损失权重，设为0表示不做类别学习
        hsv_h=0.0,               # HSV 色调抖动幅度
        hsv_s=0.05,              # HSV 饱和度抖动幅度
        hsv_v=0.05,              # HSV 明度抖动幅度
        perspective=0.0005,      # 随机透视变换强度
        degrees=1.0,            # 随机旋转角度范围（度）
        translate=0.00001,         # 随机平移比例范围
        scale=0.001,              # 随机缩放比例范围
        shear=0.0,              # 随机错切角度范围
        workers=8,              # 数据加载进程数量
        erasing=0,
        cutmix=0
    )
    #
    # # 4️⃣ 训练后推理示例
    # results = model("input/9401750994631_.pic.jpg", conf=0.25)[0]
    # print("四角点坐标(px):")   # 若装过
    # print(results.keypoints.xy)   # (1,4,2)