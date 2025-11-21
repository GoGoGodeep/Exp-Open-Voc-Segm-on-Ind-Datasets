import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from skimage.restoration import denoise_bilateral


def postprocess_preserve_small(mask,
                               small_component_thr=80,
                               hole_size_thr=200,
                               sigma_color=0.15,
                               sigma_spatial=3):
    """
    后处理：保留小目标 + 去噪
    mask: uint8 {0,255}
    """

    # 1. 二值化
    m = (mask > 128).astype(np.uint8)

    # 2. 去除极小噪点（比目标小得多）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    cleaned = np.zeros_like(m)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= small_component_thr:   # 阈值大幅减小
            cleaned[labels == i] = 1

    # 3. 填补洞（只填小洞，避免破坏内部结构）
    cleaned_inv = 1 - cleaned
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_inv)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= hole_size_thr:  # 小洞才填
            cleaned[labels == i] = 1

    # 4. 使用双边滤波进行“边缘保持”平滑
    cleaned_float = cleaned.astype(np.float32)
    smooth = denoise_bilateral(cleaned_float,
                               sigma_color=sigma_color,
                               sigma_spatial=sigma_spatial)

    smooth = (smooth > 0.4).astype(np.uint8)

    return (smooth * 255).astype(np.uint8)
