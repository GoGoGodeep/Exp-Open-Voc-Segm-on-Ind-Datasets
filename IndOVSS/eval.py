import os
import cv2
import numpy as np
from tqdm import tqdm

# ================== 路径配置 ==================
pred_dir = "/home/kexin/hd1/zkf/IndOVSS/mask_outputs"      # 预测掩码输出（.jpg）
gt_dir   = "/home/kexin/hd1/zkf/RailData/annotations/validation"  # 标注掩码（.png）

# 获取预测和标注文件名（自动匹配对应名称）
gt_names = [f for f in os.listdir(gt_dir) if f.lower().endswith(".png")]
print(f"验证集图片数量: {len(gt_names)}")

# ================== 初始化指标 ==================
intersection_total = 0
union_total = 0
correct_total = 0
pixel_total = 0

# ================== 逐张评估 ==================
for name in tqdm(gt_names):
    gt_path = os.path.join(gt_dir, name)
    pred_name = os.path.splitext(name)[0] + ".png"  # 对应预测文件
    pred_path = os.path.join(pred_dir, pred_name)

    if not os.path.exists(pred_path):
        print(f"[警告] 找不到预测文件: {pred_name}")
        continue

    # 读取图像
    gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    pred = cv2.imread(pred_path, cv2.IMREAD_COLOR)

    if gt is None or pred is None:
        print(f"[跳过] 无法读取: {name}")
        continue

    # 调整预测掩码大小一致
    if gt.shape[:2] != pred.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ground Truth 掩码：黑色=0，其余=1
    gt_mask = np.where(np.all(gt == [0, 0, 0], axis=-1), 0, 1).astype(np.uint8)

    # Predicted 掩码：白色=1，其余=0
    pred_mask = np.where(np.all(pred == [255, 255, 255], axis=-1), 1, 0).astype(np.uint8)

    # Intersection & Union
    intersection = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    union = np.logical_or(pred_mask == 1, gt_mask == 1).sum()

    # Accuracy components
    correct = (pred_mask == gt_mask).sum()
    total = gt_mask.size

    # 累积
    intersection_total += intersection
    union_total += union
    correct_total += correct
    pixel_total += total

# ================== 汇总结果 ==================
iou = intersection_total / union_total if union_total > 0 else 0
accuracy = correct_total / pixel_total if pixel_total > 0 else 0

print("\n========== 评估结果 ==========")
print(f"IoU (pantograph): {iou:.4f}")
print(f"Pixel Accuracy:   {accuracy:.4f}")
