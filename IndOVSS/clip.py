# clip_semantic_mixing.py

import torch
import open_clip  # pip install open_clip_torch

class CLIPSemanticMixer:
    """
    使用 CLIP 的 text encoder 对多个概念进行 embedding 混合：
    phrases: ["pantograph", "metal rod", "mechanical linkage", ...]
    weights: [0.5, 0.2, 0.2, 0.1]
    输出: [1, D] 的混合语义向量
    """
    def __init__(self,
                 model_name="local-dir:/home/kexin/hd1/zkf/IndOVSS/CLIP",
                 pretrained="openai",
                 device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_single(self, text: str) -> torch.Tensor:
        """
        编码单个文本，返回 [1, D] embedding（L2 归一化）
        """
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)   # [1, D]
        emb = emb.float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def encode_mix(self, phrases, weights=None) -> torch.Tensor:
        """
        phrases: List[str]
        weights: List[float] 或 None
        返回: [1, D] 的混合语义 embedding（已 L2 归一化）
        """
        assert len(phrases) > 0, "phrases 不能为空"

        if weights is None:
            weights = [1.0 / len(phrases)] * len(phrases)
        assert len(phrases) == len(weights), "phrases 和 weights 长度必须一致"

        # 编码所有短语
        tokens = self.tokenizer(phrases).to(self.device)   # [K, L]
        text_embeds = self.model.encode_text(tokens).float()  # [K, D]
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 加权混合
        w = torch.tensor(weights, device=self.device, dtype=torch.float32)  # [K]
        w = w / (w.sum() + 1e-12)
        w = w.view(-1, 1)              # [K, 1]
        mix = (w * text_embeds).sum(dim=0, keepdim=True)  # [1, D]

        # 再次归一化
        mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-12)
        return mix


if __name__ == "__main__":
    mixer = CLIPSemanticMixer(device="cuda")
    phrases = [
        "pantograph",
        "metal rod",
        "mechanical linkage",
        "carbon slider plate"
    ]
    weights = [0.5, 0.2, 0.2, 0.1]
    emb = mixer.encode_mix(phrases, weights)
    print(emb.shape)  # [1, D]
