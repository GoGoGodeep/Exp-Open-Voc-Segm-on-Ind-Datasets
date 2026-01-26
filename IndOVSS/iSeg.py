import sys
import warnings
from functools import reduce
from operator import add

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

sys.path.append("../base")

from stable_difusion import StableDiffusion
from util.miou import ShowSegmentResult

warnings.filterwarnings("ignore")

from Visualization import save_attn_grid


class iSeg(pl.LightningModule):

    def __init__(self, config, half=True):
        super().__init__()
        self.color = None
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.half = half

        self.save_hyperparameters(config.__dict__)

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",
            half=half,
            attention_layers_to_use=config.attention_layers_to_use,
        )
        if self.config.rand_seed is not None:
            self.stable_diffusion.rand_seed = self.config.rand_seed

        self.checkpoint_dir = None
        self.num_parts = self.config.num_class
        torch.cuda.empty_cache()

        # class global var
        self.cls_label = []
        self.token_sel_ids = []
        self.test_t_embedding = None

        self.showsegmentresult = ShowSegmentResult(num_classes=self.num_parts + 1)

    # ---------------------------------------------------------------------
    # 核心模块：通过 Stable Diffusion 的注意力生成掩码
    # ---------------------------------------------------------------------
    def get_masks(self, image, output_size):
        final_attention_map = torch.zeros(self.num_parts, output_size, output_size).to(self.device)
        
        cross_attention_maps, self_attention_maps = self.stable_diffusion.train_step(
            self.test_t_embedding,
            image,
            t=torch.tensor(self.config.test_t),
            generate_new_noise=True,
        )

        # 将注意力融合成最终语义掩码
        att_map, split = self.get_att_map(cross_attention_maps, self_attention_maps)
        final_attention_map[self.cls_label] += att_map
        return split, final_attention_map


    def process_cross_att(self, cross_attention_maps):
        """
        多尺度 cross-attention 融合与归一化。

        不同分辨率 (8×8, 16×16, 32×32, 64×64) 的 cross-attention 层
        在空间细节与语义层级上具有互补性，因此通过加权融合获得综合注意力。
        """
        weight_layer = {8: 0.0, 16: 0.5, 32: 0.5, 64: 0.0}
        cross_attention = []

        for key, values in cross_attention_maps.items():
            if len(values) == 0:
                continue
            
            print(type(values))

            # 对 batch 与 head 维度求均值
            values = values.mean(1)
            
            # 每个像素位置归一化
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)

            # 对低分辨率层插值到统一 64×64 尺寸
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode="bilinear", align_corners=False)

            # # === 加上可视化保存 ===
            # save_attn_grid(normed_attn, self.token_sel_ids, f"/home/kexin/hd1/zkf/IndOVSS/att_vis/{key}.png")
            
            # 加权融合
            cross_attention.append(weight_layer[key] * normed_attn)

        # 按层堆叠后求和，flatten 为 (H*W, token)
        cross_attention = torch.stack(cross_attention, dim=0).sum(0)[0]     # [tokens, 64, 64]

        cross_attention = cross_attention.flatten(-2, -1).permute(1, 0)     # [4096, tokens]

        # 根据 token 选择索引聚合到目标类别维度
        cross_attention = torch.stack(
            [cross_attention[:, sel].mean(1) for sel in self.token_sel_ids], dim=1
        )

        return cross_attention[None]


    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)


    def get_text_embedding(self, text: str) -> torch.Tensor:
        text_input = self.stable_diffusion.tokenizer(
            text,
            padding="max_length",
            max_length=self.stable_diffusion.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.set_grad_enabled(False):
            embedding = self.stable_diffusion.text_encoder(text_input.input_ids.cuda(), output_hidden_states=True)[0]
            embedding = embedding.half() if self.half else embedding
        return embedding

    def test_step(self, batch, batch_idx):
        print("step of test")
        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("end of test.")


    @staticmethod
    def get_boundry_and_eroded_mask(mask):
        """
        对每个类别掩码进行腐蚀操作并提取边界区域。
        """
        kernel = np.ones((7, 7), np.uint8)
        eroded_mask = np.zeros_like(mask)
        boundry_mask = np.zeros_like(mask)
        for part_mask_idx in np.unique(mask)[1:]:
            part_mask = np.where(mask == part_mask_idx, 1, 0)
            part_mask_erosion = cv2.erode(part_mask.astype(np.uint8), kernel, iterations=1)
            part_boundry_mask = part_mask - part_mask_erosion
            eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
            boundry_mask = np.where(part_boundry_mask > 0, part_mask_idx, boundry_mask)
        return eroded_mask, boundry_mask


    @staticmethod
    def get_colored_segmentation(mask, boundry_mask, image, colors):
        boundry_mask_rgb = 0
        if boundry_mask is not None:
            boundry_mask_rgb = torch.repeat_interleave(boundry_mask[None, ...], 3, 0).type(torch.float)
            for j in range(3):
                for i in range(1, len(colors) + 1):
                    boundry_mask_rgb[j] = torch.where(
                        boundry_mask_rgb[j] == i, colors[i - 1][j] / 255, boundry_mask_rgb[j]
                    )
        mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
        for j in range(3):
            for i in range(1, len(colors) + 1):
                mask_rgb[j] = torch.where(mask_rgb[j] == i, colors[i - 1][j] / 255, mask_rgb[j])
        if boundry_mask is not None:
            final = torch.where(
                boundry_mask_rgb + mask_rgb == 0, image, boundry_mask_rgb * 0.7 + mask_rgb * 0.5 + image * 0.3
            )
            return final.permute(1, 2, 0)
        else:
            final = torch.where(mask_rgb == 0, image, mask_rgb * 0.6 + image * 0.4)
            return final.permute(1, 2, 0)


class interaction_iSeg(iSeg):
    def __init__(self, configs, half=True):
        super().__init__(config=configs, half=half)
        self.self_attn = None
        self.cross_attn = None
        self.img = None

    def get_att_map(self, cross_attention_maps, self_attention_maps):
        # cross attention 特征归一化 & 上采样融合
        cross_attn = self.process_cross_att(cross_attention_maps).float()
        cross_attn = cross_attn - cross_attn.amin(dim=-2, keepdim=True)
        cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)

        aff_mat = self_attention_maps[64].mean(1).flatten(-2, -1).clone()
        aff_mat = aff_mat.permute(0, 2, 1).float()      # [1, 4096, 4096]

        self.cross_attn = cross_attn
        self.self_attn = aff_mat
        return None

    def get_masks(self, image, output_size, is_use_ers=True):
        """
        调用 Stable Diffusion 提取注意力但不生成最终掩码。
        """
        cross_attention_maps, self_attention_maps = self.stable_diffusion.train_step(
            self.test_t_embedding,
            image,
            t=torch.tensor(self.config.test_t),
            generate_new_noise=True,
        )
        self.get_att_map(cross_attention_maps, self_attention_maps)
        return None


    def test_step(self, batch, batch_idx):
        """
        交互模式的 test_step：
        - 获取输入图像与文本；
        - 编码文本；
        - 增强关键 token 的 embedding；
        - 提取并缓存注意力矩阵。
        """
        self.img, text_captions, self.token_sel_ids = batch
        image = self.img.half() if self.half else self.img

        # 1. 原始文本编码（SD 文本 encoder）
        self.test_t_embedding = self.get_text_embedding(text_captions)

        # 2. 计算语义 token 索引
        meaning_index = reduce(add, self.token_sel_ids)

        self.test_t_embedding[:, meaning_index] *= self.config.enhanced

        self.get_masks(image, self.config.test_mask_size)

        return None