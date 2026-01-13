import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math


def save_attn_grid(normed_attn, token_sel_ids, save_path):
    """
    normed_attn: Tensor [Tokens, H, W]
    token_sel_ids: [4, 5, 6] class name 对应的层数，即token数
    """
    normed_attn = normed_attn.squeeze(0)   # 去掉 batch 维度    [77, 64, 64]
    # print(normed_attn.shape)

    normed_attn = torch.stack([
        normed_attn[sel].mean(0)   # mean(0) -> [H, W]
        for sel in token_sel_ids
    ])
    # print(normed_attn.shape)      # [1, 64, 64]

    T, H, W = normed_attn.shape
    cols = int(math.ceil(math.sqrt(T)))
    rows = int(math.ceil(T / cols))

    canvas = torch.zeros((rows * H, cols * W))

    for i in range(T):
        r, c = i // cols, i % cols
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = normed_attn[i]

    plt.figure(figsize=(5, 5))
    plt.imshow(canvas.cpu(), cmap='viridis')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
