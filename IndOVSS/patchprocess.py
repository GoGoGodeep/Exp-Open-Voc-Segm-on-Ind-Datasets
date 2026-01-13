import numpy as np


def split_image_patches(np_img, rows=3, cols=3, overlap=0):
    """
    将 numpy 图片分割成 (rows x cols) 个 patch。
    overlap: 重叠像素数（水平和垂直相同），整数 >= 0。
    返回列表 of (patch, (y0, y1, x0, x1))
    坐标为 patch 在原图上的包含范围 [y0:y1, x0:x1)（整型）
    """
    H, W = np_img.shape[:2]
    # 每个非重叠单元尺寸（向下取整）
    base_h = H // rows
    base_w = W // cols

    patches = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * base_h
            x0 = c * base_w
            # 计算默认 y1,x1（不超过末尾）
            y1 = (r + 1) * base_h if r < rows - 1 else H
            x1 = (c + 1) * base_w if c < cols - 1 else W

            # 扩展 overlap（左右/上下），并裁剪到图像边界
            y0_ext = max(0, y0 - overlap)
            x0_ext = max(0, x0 - overlap)
            y1_ext = min(H, y1 + overlap)
            x1_ext = min(W, x1 + overlap)

            patch = np_img[y0_ext:y1_ext, x0_ext:x1_ext].copy()
            patches.append((patch, (y0_ext, y1_ext, x0_ext, x1_ext)))

    return patches


def stitch_patches_to_image(patch_masks_with_coords, image_shape):
    """
    将 patch 的 mask（单通道 float or uint8）拼回原始图像尺寸。
    利用加权累加并最终归一化，避免缝隙与硬拼接。
    """
    if len(image_shape) == 3:
        H, W, _ = image_shape
    else:
        H, W = image_shape

    acc = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for mask_np, (y0, y1, x0, x1) in patch_masks_with_coords:
        # convert to float [0,1]
        m = mask_np.astype(np.float32)
        if m.max() > 1.0:
            m = m / 255.0

        ph, pw = m.shape[:2]
        # 构造融合权重：用二维三角窗（线性权重）使得 patch 中心权重大，边缘权重小，平滑缝隙
        wy = (np.linspace(0, 1, ph) + np.linspace(0, 1, ph)[::-1]) / 2.0  # symmetric
        wx = (np.linspace(0, 1, pw) + np.linspace(0, 1, pw)[::-1]) / 2.0
        wy = wy / wy.max()
        wx = wx / wx.max()
        window = np.outer(wy, wx)  # shape (ph, pw) in [0,1]
        # normalize window to avoid zeros
        window = window + 1e-6

        acc[y0:y1, x0:x1] += m * window
        weight[y0:y1, x0:x1] += window

    # 防止除零
    weight = np.where(weight == 0, 1e-6, weight)
    stitched = acc / weight
    stitched = np.clip(stitched, 0.0, 1.0)
    stitched_u8 = (stitched * 255.0).astype(np.uint8)

    return stitched_u8