'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode):

        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device)

        if mode == 'image':
            # return image features
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text.input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
            return output.last_hidden_state


class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 ):
        """
        多模态图像描述生成模型（解码器部分）
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        self.tokenizer = init_tokenizer()

        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width

        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 文本token化处理
        text = self.tokenizer(
            caption, padding='longest',
            truncation=True, max_length=40,
            return_tensors="pt").to(image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # 构建解码目标（忽略提示词和padding部分的损失）
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss   # 即bert mlm损失
        return loss_lm

    def generate(
            self, image, sample=False,
            num_beams=3, max_length=30,
            min_length=10, top_p=0.9,
            repetition_penalty=1.0,
    ):
        """图像描述生成接口（捕获每个token的注意力）"""
        image_embeds = self.visual_encoder(image)

        # -----------------------------------------------------------------
        num_layers = len(self.text_decoder.base_model.base_model.encoder.layer)
        self.attentions = [[] for _ in range(num_layers)]

        # 定义钩子函数（用于捕获跨模态注意力）
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                attention_weights = output[1][1]  # 获取实际注意力矩阵
                self.attentions[layer_idx].append(attention_weights.detach().cpu())
            return hook_fn

        hooks = []
        for layer_idx, layer in enumerate(self.text_decoder.base_model.base_model.encoder.layer):
            hook = layer.crossattention.self.register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)
        # -----------------------------------------------------------------

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # 核采样（多样性生成）
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # Beam Search（确定性生成）
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])     # 去除提示词部分

        # -----------------------------------------------------------------
        # 生成完成后移除钩子
        for hook in hooks:
            hook.remove()

        # 获取生成的token列表（第一个beam）
        output_ids = outputs[0].cpu().numpy().tolist()
        output_tokens = self.tokenizer.convert_ids_to_tokens(
            output_ids,
            skip_special_tokens=True
        )

        # 生成并保存每个token的注意力图
        self._save_attention_map(
            image,
            captions,
            output_tokens=output_tokens,  # 传递token列表
            beam_index=0
        )
        # -----------------------------------------------------------------

        return captions

    def _save_attention_map(self, image, captions, output_tokens, beam_index=0):
        # 定义标准化参数
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # 进行反标准化处理，否则图片的颜色会与原来不一致
        def denormalize(tensor):
            tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            tensor = tensor * std + mean  # 反标准化
            tensor = np.clip(tensor, 0, 1)  # 裁剪到合理范围
            return tensor

        # 获取反标准化后的图像
        denorm_img = denormalize(image[0])  # 处理第一个样本

        """生成并保存每个token的注意力热力图"""
        num_layers = len(self.attentions)
        all_layer_attentions = []

        # 提取指定beam的注意力数据
        for layer_idx in range(num_layers):
            layer_attn = torch.stack(self.attentions[layer_idx])    # layer_attn形状: [生成步数, batch, 注意力头数, 目标长度, 源长度]
            layer_attn = layer_attn[:, beam_index, ...]  # 选择特定beam
            all_layer_attentions.append(layer_attn)

        # 使用最后一层最后一个注意力头的权重
        last_layer_attn = all_layer_attentions[-1]  # [steps, heads, tgt_len, src_len]  [10, 12, 577, 64]

        for step_idx in range(len(output_tokens)):
            token = output_tokens[step_idx]

            # 获取当前步骤的注意力权重（最后一层，所有头的平均  ??? 平均的做法有待商榷）
            # 步骤step_idx对应tgt_len=step_idx+1，取最后一个位置step_idx
            step_attn = last_layer_attn[:, step_idx, :, step_idx]  # [heads, src_len]
            attn_weights = step_attn.mean(dim=0)  # [src_len]

            # 去除CLS token（假设第一个位置）
            attn_weights = attn_weights[1:]

            # 转换为2D图
            patch_size = int(np.sqrt(attn_weights.shape[0]))
            attn_map = attn_weights.reshape(patch_size, patch_size).cpu().numpy()

            # 上采样到原图尺寸
            attn_map = F.interpolate(
                torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float(),
                size=(image.shape[2], image.shape[3]),
                mode='bilinear'
            ).squeeze().numpy()

            # 归一化
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            # 可视化
            plt.figure(figsize=(12, 6))

            # 修改图像显示部分：
            # 左图：原始图像（反标准化后）
            plt.subplot(1, 2, 1)
            plt.imshow(denorm_img)
            plt.axis('off')
            plt.title('ori img')

            # 右图：当前token的注意力
            plt.subplot(1, 2, 2)
            plt.imshow(denorm_img)
            plt.imshow(attn_map, alpha=0.5, cmap='jet')
            plt.axis('off')
            plt.title(f'Token {step_idx}: {token}\ngenerated captions: {captions[0]}')

            # 保存
            os.makedirs('attention_maps_per_token', exist_ok=True)
            plt.savefig(
                f'attention_maps_per_token/step{step_idx}_{token}.png',
                bbox_inches='tight',
                dpi=150
            )
            plt.close()


def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def blip_feature_extractor(pretrained='', **kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg
