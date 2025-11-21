import argparse
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from postprocess import postprocess_preserve_small

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def load_model(device, use_half=False):
    # Import same modules as GUI
    from ui.arguments import init_args
    from iSeg import interaction_iSeg

    cfg = init_args()
    model = interaction_iSeg(cfg)

    # 将模型移动到目标设备
    model = model.to(device)

    # 如果要半精度推理
    if use_half:
        for param in model.parameters():
            param.data = param.data.half()

    # 与 GUI 相同：执行测试前初始化
    model.on_test_start()

    return model


def img_to_tensor(np_img, device, half=False):
    if np_img.dtype == np.uint8:
        img = torch.tensor(np_img.astype(np.float32) / 255.0, device=device)
    else:
        img = torch.tensor(np_img.astype(np.float32), device=device)
    img = img.half() if half else img
    img = img.permute(2, 0, 1)[None]
    return img


def build_text_prompt_indices(model, text: str):
    """
    构建文本提示的 token 索引映射。

    作用：
        从文本提示中提取出每个子句对应的 token 位置索引（sel_idx），
        供 Stable Diffusion 的 cross-attention 层使用，以便在注意力中
        只聚焦特定语义片段。
    """
    text = text.strip()
    tokenizer = model.stable_diffusion.tokenizer

    mask = tokenizer(text.split(";"), padding="max_length").attention_mask
    mask = [sum(m) - 2 for m in mask]

    sel_idx, start_ids = [], []
    for idx, l in enumerate(mask):
        # 这里 4 是根据 stable diffusion 提示模板：
        #   "a photo of {object}"，前面通常有 4 个固定 token
        FIRST_OBJECT_OFFSET = 4
        start_ids.append(start_ids[idx - 1] + 1 + mask[idx] if idx > 0 else FIRST_OBJECT_OFFSET)
        sel_idx.append([start_ids[-1] + sel_id for sel_id in range(mask[idx])])

    text = text.replace(";", " and ")
    return text, sel_idx


def run_one_image(model, np_img, iter_count=10, thr=0.5, ent=0.5, device=None):
    """
    使用文本提示 "pantograph" 进行推理，输出二分类掩码。
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    img_tensor = img_to_tensor(np_img, device=device, half=model.half)

    # === Step 1. 准备文本提示 ===
    text = "pantograph"
    txt, sel_idx = build_text_prompt_indices(model, text)
    print("Processed text:", txt)
    print("sel_idx:", sel_idx)
    print("tokens:", model.stable_diffusion.tokenizer.tokenize(txt))

    # 可选：更详细的语义提示，可提高 cross-attention 的语义聚焦性
    # prompt = "The pantograph is a large, articulated mechanical arm mounted on the roof of the vehicle, extending upward to make contact with overhead power lines."
    # prompt = "The pantograph is a mechanical device with a vertical support structure and a horizontal arm extending upward"
    # prompt_1 = "The pantograph consists of two bent, bow-shaped components."
    promot_1_e = "The pantograph consists of a curved carbon plate and a metal frame with mounting brackets."
    # prompt_2 = "The pantograph consists of two curved, angled, bow-shaped components."
    # prompt = f"a photograph of {text} and other object and background"
    # prompt = f"a photograph of {text} and background"
    # prompt_blip = "a photograph of {text} being used in a warehouse"
    
    with torch.no_grad():
        # === Step 2. 模型前向推理，捕获 self / cross attention ===
        model.test_step((img_tensor, promot_1_e, sel_idx), 0)

        # === Step 3. 提取并预处理注意力特征 ===
        self_attn = model.self_attn.clone()
        cross_attn = model.cross_attn.clone()

        # cross-attention 张量变换到统一形状
        cross_attn = cross_attn.permute(0, 2, 1).reshape(1, -1, 64, 64)
        # 归一化到 [0,1]
        cross_attn -= cross_attn.amin(dim=(-2, -1), keepdim=True)
        cross_attn /= cross_attn.amax(dim=(-2, -1), keepdim=True)

        # 展平以便后续矩阵乘法传播
        cross_attn_proc = cross_attn.permute(0, 2, 3, 1).reshape(1, 4096, -1)
        cross_attn_proc = cross_attn_proc - cross_attn_proc.amin(dim=-2, keepdim=True)
        cross_attn_proc = cross_attn_proc / (cross_attn_proc.sum(dim=-2, keepdim=True) + 1e-12)

        # self-attention 归一化与熵增强
        self_attn = self_attn / (self_attn.sum(dim=-1, keepdim=True) + 1e-12)
        self_attn = self_attn / (torch.amax(self_attn, dim=-2, keepdim=True) + 1e-12)
        self_attn = self_attn + torch.where(
            self_attn == 0, torch.zeros_like(self_attn),
            ent * (torch.log10(torch.e * self_attn))
        )
        self_attn = torch.clamp(self_attn, 0, 1)
        self_attn = self_attn / (self_attn.sum(dim=-1, keepdim=True) + 1e-12)

        # === Step 4. 迭代传播 self → cross 注意力 ===
        # 多轮传播使目标区域聚焦收敛
        for _ in range(iter_count):
            cross_attn_proc = torch.bmm(self_attn, cross_attn_proc)
            cross_attn_proc = cross_attn_proc - cross_attn_proc.amin(dim=-2, keepdim=True)
            cross_attn_proc = cross_attn_proc / (cross_attn_proc.sum(dim=-2, keepdim=True) + 1e-12)

        # === Step 5. 恢复空间结构并生成掩码 ===
        cross_attn_proc = cross_attn_proc / (cross_attn_proc.amax(dim=-2, keepdim=True) + 1e-12)
        cross_attn_proc = cross_attn_proc.mean(-1, keepdim=True)
        cross_attn_proc = cross_attn_proc / (cross_attn_proc.amax(dim=-2, keepdim=True) + 1e-12)
        cross_attn_proc = cross_attn_proc.permute(0, 2, 1).reshape(1, -1, 64, 64)

        # 插值回原图尺寸
        H, W = np_img.shape[:2]
        cross_attn_up = F.interpolate(cross_attn_proc, size=(H, W), mode='bilinear', align_corners=False)[0]
        
        # 根据阈值生成二值掩码
        thr_channel = torch.ones_like(cross_attn_up[[0], :, :]) * float(thr)
        mask_stack = torch.cat((thr_channel, cross_attn_up), dim=0)
        mask = mask_stack.argmax(dim=0).squeeze().cpu().numpy().astype(np.uint8)
        binary = (mask != 0).astype(np.uint8) * 255

        binary = postprocess_preserve_small(binary)

        # # === Step 6. 可视化 cross-attention 热力图（调试用） ===
        # attn_map = cross_attn_up[0].detach().cpu().numpy()  # shape: (H, W)
        # attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-12)

        # plt.figure(figsize=(6, 6))
        # plt.imshow(attn_map_norm, cmap='jet')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig("attention_vis.png", bbox_inches='tight', pad_inches=0)
        # plt.close()
        # print('Success!')

        return binary


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


def process_image_patches(np_img, model, rows=2, cols=2, overlap=0,
                          run_args=None):
    """
    将输入图片 np_img 拆成 rows×cols 个 patch（可带重叠），
    分别送入 run_one_image(model, patch, ...)，
    最终拼接回原图。
    保证输出：
        - 尺寸与输入一致
        - 仅包含黑白（0 与 255）两种像素值
    """
    H, W = np_img.shape[:2]
    run_args = {} if run_args is None else run_args.copy()
    ph = H // rows
    pw = W // cols

    stitched = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            # ---- 计算 patch 范围（带重叠）----
            y0 = max(0, r * ph - overlap)
            y1 = min(H, (r + 1) * ph + overlap if r < rows - 1 else H)
            x0 = max(0, c * pw - overlap)
            x1 = min(W, (c + 1) * pw + overlap if c < cols - 1 else W)

            patch = np_img[y0:y1, x0:x1]

            # # 改进：将原图切 patch 后，每个 patch 缩放成 512×512 再送入 SD
            # patch = Image.fromarray(patch).resize((256,256), Image.BICUBIC)

            # ---- 单 patch 推理 ----
            mask = run_one_image(
                model, patch,
                iter_count=run_args.get("iter_count", 5),
                thr=run_args.get("thr", 0.5),
                ent=run_args.get("ent", 0.5),
                device=run_args.get("device", None),
            )
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(np.float32)
            if mask.max() > 1:
                mask /= 255.0

            # ---- 融合权重（平滑边界）----
            h, w = mask.shape
            wy = np.linspace(0, 1, h)
            wx = np.linspace(0, 1, w)
            window = np.outer(np.minimum(wy, wy[::-1]),
                              np.minimum(wx, wx[::-1])) + 1e-6

            stitched[y0:y1, x0:x1] += mask * window
            weight[y0:y1, x0:x1] += window

    # ---- 融合并归一化 ----
    weight[weight == 0] = 1e-6
    merged = stitched / weight
    merged = np.clip(merged, 0, 1)

    # ---- 二值化（仅保留 0 / 255）----
    threshold = 0.5  # 可根据任务调整
    binary = (merged >= threshold).astype(np.uint8) * 255

    # ---- 强制校验尺寸一致 ----
    binary = np.array(Image.fromarray(binary).resize((W, H), Image.NEAREST))
    assert binary.shape == (H, W), f"Size mismatch: got {binary.shape}, expected {(H, W)}"

    return binary


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    ensure_dir(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()
    print(f"Using device: {device}, half precision: {use_half}")

    model = load_model(device=device, use_half=use_half)

    exts = ('*.jpg', '*.jpeg', '*.png')
    files = []
    for ext in exts:
        files += glob(os.path.join(input_dir, ext))
    files = sorted(files)
    if len(files) == 0:
        print("No images found in", input_dir)
        return

    print(f"Found {len(files)} images. Running inference...")

    # for p in tqdm(files):
    #     try:
    #         img = Image.open(p).convert('RGB')#.resize((320, 240))
    #         np_img = np.array(img)
    #         binary = run_one_image(model, np_img,
    #                                iter_count=args.iter,
    #                                thr=args.thr,
    #                                ent=args.ent,
    #                                device=device)
    #         base = os.path.splitext(os.path.basename(p))[0]
    #         out_path = os.path.join(output_dir, base + '.png')
    #         Image.fromarray(binary).save(out_path)
    #     except Exception as e:
    #         print(f"Failed on {p}: {e}")

    # 裁剪图片
    for p in tqdm(files):
        try:
            img = Image.open(p).convert('RGB')
            np_img = np.array(img)
            H, W = np_img.shape[:2]

            # ---- 判断是否进行 patch 处理 ----
            if H < 256 or W < 256:
                # 尺寸较小，直接整图推理
                binary = run_one_image(
                    model, np_img,
                    iter_count=args.iter,
                    thr=args.thr,
                    ent=args.ent,
                    device=device,
                )
                if binary.ndim == 3:
                    binary = binary[..., 0]
                if binary.max() <= 1:
                    binary = (binary * 255).astype(np.uint8)
                else:
                    binary = binary.astype(np.uint8)
            else:
                # 尺寸较大，使用 patch 处理
                stitched_mask = process_image_patches(
                    np_img, model,
                    rows=2, cols=2, overlap=0,
                    run_args={
                        "iter_count": args.iter,
                        "thr": args.thr,
                        "ent": args.ent,
                        "device": device,
                    }
                )
                binary = stitched_mask

            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(output_dir, base + '.png')
            Image.fromarray(binary).save(out_path)

        except Exception as e:
            print(f"Failed on {p}: {e}")

    print("Done. Masks saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless iSeg inference for pantograph segmentation")
    parser.add_argument("--input_dir", type=str,
                        default="/home/kexin/hd1/zkf/RailData/images/validation")
    parser.add_argument("--output_dir", type=str,
                        default="/home/kexin/hd1/zkf/IndOVSS/mask_outputs")
    parser.add_argument("--iter", type=int, default=5, help="Iteration count")
    parser.add_argument("--thr", type=float, default=0.5, help="Threshold 0..1")
    parser.add_argument("--ent", type=float, default=0.5, help="Entropy scaling factor")
    args = parser.parse_args()
    main(args)


# if __name__ == "__main__":
#     # ========== 参数设定 ==========
#     img_path = "/home/kexin/hd1/zkf/IndOVSS/imgs/cut.jpg"  # 你要可视化的图片
#     save_path = "/home/kexin/hd1/zkf/IndOVSS/attention_vis.png"  # 注意力图保存路径

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     use_half = torch.cuda.is_available()
#     print(f"Using device: {device}, half precision: {use_half}")

#     # === 加载模型 ===
#     print("Loading iSeg model with text prior 'pantograph'...")
#     model = load_model(device=device, use_half=use_half)

#     # === 读取图像 ===
#     from PIL import Image

#     img = Image.open(img_path).convert('RGB')
#     np_img = np.array(img)

#     # === 推理 ===
#     binary = run_one_image(model, np_img,
#                            iter_count=5, thr=0.5, ent=0.5,
#                            device=device)
