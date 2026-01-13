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

from clip import CLIPSemanticMixer


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

            # 对 batch 与 head 维度求均值
            values = values.mean(1)
            
            # 每个像素位置归一化
            normed_attn = values / values.sum(dim=(-2, -1), keepdim=True)

            # 对低分辨率层插值到统一 64×64 尺寸
            if key != 64:
                normed_attn = F.interpolate(normed_attn, size=(64, 64), mode="bilinear", align_corners=False)

            # === 加上可视化保存 ===
            save_attn_grid(normed_attn, self.token_sel_ids, f"/home/kexin/hd1/zkf/IndOVSS/att_vis/{key}.png")
            
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


    def get_att_map(self, cross_attention_maps, self_attention_maps):
        cross_att = self.process_cross_att(cross_attention_maps).float()
        
        # self-attention: [B, H*W, H*W]
        self_att = self_attention_maps[64].reshape(-1, 64 * 64, 64 * 64).permute(0, 2, 1).float()
        
        # self-attn 扩散 cross-attn 的语义权重
        cross_att = torch.bmm(self_att, cross_att)
        
        # 还原空间结构并插值至目标尺寸
        att_map = cross_att.unflatten(dim=-2, sizes=(64, 64)).permute(0, 3, 1, 2)
        att_map = F.interpolate(att_map, size=self.config.test_mask_size, mode="bilinear", align_corners=False)
        att_map = att_map[0]
        
        # 归一化 [0,1]
        att_map -= att_map.amin(dim=(-2, -1), keepdim=True)
        att_map /= att_map.amax(dim=(-2, -1), keepdim=True)
        return att_map, None


    def show_cam_on_image(self, mask, save_path):
        mask = np.uint8(255 * mask.cpu())
        mask = cv2.resize(mask, dsize=(self.config.patch_size, self.config.patch_size))
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cv2.imwrite(save_path, cam)


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

        # ✅【新增】CLIP 语义混合相关
        self.use_clip_mix = getattr(configs, "use_clip_mix", False)
        self.clip_lambda = getattr(configs, "clip_lambda", 0.2)  # CLIP 与原 embedding 的混合权重
        self.clip_mixer = None
        self.clip_adapter = None  # 若 CLIP 维度与 SD 文本维度不同，用线性层适配

        if self.use_clip_mix:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.clip_mixer = CLIPSemanticMixer(
                model_name=getattr(configs, "clip_model_name", "local-dir:/home/kexin/hd1/zkf/IndOVSS/CLIP"),
                pretrained=getattr(configs, "clip_pretrained", "openai"),
                device=device
            )


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


    def show_cam_on_image(self, mask, type_=cv2.COLORMAP_JET):
        mask = np.uint8(255 * mask.cpu())
        heatmap = cv2.applyColorMap(mask, type_)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap / np.max(heatmap)
        return np.uint8(255 * cam)[:, :, [2, 1, 0]]


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
    

    # ✅【新增】构造 CLIP 混合 embedding，并对齐到 SD 文本空间
    def build_clip_mixed_embedding(self, text_captions):
        """
        text_captions: 原始文本提示（可以是字符串或 list[str]）
        返回: [1, D_text]，已对齐到 Stable Diffusion 文本 encoder 的维度
        """
        if not self.use_clip_mix or self.clip_mixer is None:
            return None

        # 1. 准备语义短语：未知类 + 若干已知类（可以从 config 里读）
        #   这里默认是“受电弓”场景，你可以在 configs 里自定义
        main_caption = text_captions[0] if isinstance(text_captions, (list, tuple)) else text_captions
        phrases = getattr(
            self.config,
            "clip_mix_phrases",
            # [
            #     main_caption,                 # e.g. "pantograph"
            #     "train pantograph head",
            #     "pantograph carbon slider",
            #     "pantograph upper frame"
            # ]
            [
                "cut ",                 # e.g. "cut"
                "scratches",
                "nick",
                "scar"
            ]
        )
        weights = getattr(
            self.config,
            "clip_mix_weights",
            [0.6, 0.2, 0.1, 0.1]
        )

        # 2. CLIP 空间中进行语义混合
        clip_mix_emb = self.clip_mixer.encode_mix(phrases, weights)  # [1, D_clip]

        # 3. 对齐到 SD 文本 embedding 维度
        with torch.no_grad():
            # 临时拿一遍原始 embedding 看维度
            base_emb = self.get_text_embedding(text_captions)  # [B, L, D_text]
            _, _, D_text = base_emb.shape
            D_clip = clip_mix_emb.shape[-1]

            device = base_emb.device
            clip_mix_emb = clip_mix_emb.to(device)

            if D_clip != D_text:
                # 第一次使用时创建 adapter
                if self.clip_adapter is None:
                    self.clip_adapter = torch.nn.Linear(D_clip, D_text).to(device)
                clip_mix_emb = self.clip_adapter(clip_mix_emb)  # [1, D_text]

            # 归一化一下（可选）
            clip_mix_emb = clip_mix_emb / (clip_mix_emb.norm(dim=-1, keepdim=True) + 1e-12)

        return clip_mix_emb  # [1, D_text]


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

        # 3. ✅ 如果开启 CLIP 语义混合，则把混合后的 embedding 注入到这些 token 上
        if self.use_clip_mix:
            print("use_clip_mix!")
            
            clip_mix_emb = self.build_clip_mixed_embedding(text_captions)  # [1, D_text] or None
            if clip_mix_emb is not None:
                B, L, D = self.test_t_embedding.shape

                # 扩展成 [B, D]，便于写入
                clip_mix_emb_bd = clip_mix_emb.expand(B, -1)  # [B, D]

                # 对每个语义 token 位置做线性混合：
                lambda_ = float(self.clip_lambda)

                for idx in meaning_index:
                    if idx < 0 or idx >= L:
                        continue
                    self.test_t_embedding[:, idx, :] += lambda_ * clip_mix_emb_bd

        # 4. （可选）仍然保留你原来的 “增强系数” 放大
        self.test_t_embedding[:, meaning_index] *= self.config.enhanced

        # 5. 提取注意力特征
        self.get_masks(image, self.config.test_mask_size)

        return None