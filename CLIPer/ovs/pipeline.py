from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_model.stable_diffusion import diffusion
from modified_clip.model import CLIPer, CLIPer_BLIP
from modified_clip.open_model import OpenCLIPer

from transformers import BlipProcessor, BlipForConditionalGeneration

blip_processor = BlipProcessor.from_pretrained(
    r"/home/saki/Desktop/DiffSegmenter/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    r"/home/saki/Desktop/DiffSegmenter/blip-image-captioning-large").to("cuda")
blip_model.eval()


class Pipeline(nn.Module):
    def __init__(self, cfg, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.cliper, self.attn_refine = None, None
        self.device = device
        if cfg.model_name in ["ViT-B/16", "ViT-L/14", 'RN101']:
            self.cliper = CLIPer(model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
                                 fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
            # self.cliper = CLIPer_BLIP(model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
            #                           fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
        elif cfg.model_name == "ViT-H/14":
            self.cliper = OpenCLIPer(model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
                                     fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
        else:
            raise NotImplementedError("Unknown Model")

        if hasattr(cfg, "refinement") and cfg.refinement in ["SFSA", "mean", "selection"]:
            self.attn_refine = diffusion(attention_layers_to_use=cfg.attention_layers_to_use,
                                         model=cfg.sd_version, time_step=cfg.time_step, device=self.device,
                                         dtype=torch.float16 if "cuda" in str(self.device) else torch.float32)

        self.cfg = cfg

    def refinement(self, ori_img, pred_mask, classes_name, blip_img):
        import os
        import matplotlib.pyplot as plt

        debug_dir = f"./debug_results/{hash(ori_img)}/"  # 使用图像哈希值创建唯一目录
        os.makedirs(debug_dir, exist_ok=True)

        # 保存原始输入可视化
        def _save_tensor(tensor, filename, cmap='viridis', vmin=None, vmax=None):
            tensor = tensor.detach().cpu().float()
            if tensor.ndim == 3:  # CHW格式转HWC
                tensor = tensor.permute(1, 2, 0) if tensor.shape[0] in [1, 3] else tensor.mean(0)
            plt.imshow(tensor, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis('off')
            plt.savefig(os.path.join(debug_dir, filename), bbox_inches='tight', pad_inches=0)
            plt.close()

        if self.attn_refine is not None:
            pred_mask = F.interpolate(pred_mask[None], size=(64, 64), mode='bilinear',
                                      align_corners=False)[0].flatten(-2).float()
            cross_att = pred_mask.transpose(0, 1)
            _save_tensor(pred_mask.reshape(-1, 64, 64), f"cross_att.png")

            combined_self_att = None
            for idx, classname in enumerate(classes_name):
                # blip生成增强文本描述
                text = f"a photograph of {classname}"

                # 使用BLIP进行文本-图像输入处理
                inputs = blip_processor(blip_img, text, return_tensors="pt").to(self.device)

                # 使用BLIP生成新的文本描述，进行增强
                generation_args = {
                    "max_length": 30,
                    "num_beams": 5,
                    "repetition_penalty": 2.0
                }
                out = blip_model.generate(**inputs, **generation_args)
                generated_text = blip_processor.decode(out[0], skip_special_tokens=True)
                print(f'class_name: {classname} blip_text: {generated_text}')

                self.attn_refine(ori_img.to(self.device), generated_text)
                self_att = torch.cat(
                    [self.attn_refine.attention_maps[idx][0] for idx in self.cfg.attention_layers_to_use]).float()

                # 归一化处理
                self_att /= torch.amax(self_att, dim=-2, keepdim=True) + 1e-5
                self_att = torch.where(self_att < 0.1, 0, self_att)
                self_att /= self_att.sum(dim=-1, keepdim=True) + 1e-5

                if self.cfg.refinement == "mean":
                    self_att = self_att.mean(0)
                elif self.cfg.refinement == "selection":
                    self_att = self_att[self.cfg.attention_idx]
                else:
                    self_att = reduce(torch.matmul, self_att, torch.eye(self_att.shape[-1], device=self_att.device))

                # 可视化单类别注意力
                _save_tensor(self_att.mean(0).reshape(64, 64), f"{idx}_{classname}_self_att.png")
                _save_tensor(self_att, f"{idx}_{classname}_self_att_layers.png")

                # 合并多个类名的self_att
                if combined_self_att is None:
                    combined_self_att = self_att
                else:
                    combined_self_att += self_att

            combined_self_att /= len(classes_name)
            _save_tensor(combined_self_att, "combined_self_attention.png")

            pred_mask = (combined_self_att @ cross_att).transpose(0, 1).reshape(-1, 64, 64)
            _save_tensor(pred_mask, "final_pred_mask.png")

        return pred_mask

    def forward(
            self,
            ori_img, img,
            classify_fg_text_features, classify_bg_text_features,
            classes_name, blip_img
    ):
        segment_results = self.cliper(img, classify_fg_text_features, classify_bg_text_features)
        seg = segment_results["seg"]

        final_score = seg.amax(dim=0)
        seg = seg.transpose(0, 1).reshape(-1, self.cliper.img_h, self.cliper.img_w)

        pred_mask = self.refinement(ori_img, seg, classes_name, blip_img)

        final_score = pred_mask.amax(dim=(-1, -2)) * 0.5 + final_score * 0.5

        return pred_mask, final_score
