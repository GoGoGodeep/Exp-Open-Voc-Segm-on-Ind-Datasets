import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


processor = BlipProcessor.from_pretrained("/home/kexin/hd1/zkf/DiffSegmenter/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("/home/kexin/hd1/zkf/DiffSegmenter/blip-image-captioning-large").to("cuda")


def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


with torch.no_grad():
    same_seeds(3407)
    
    img_path = "/home/kexin/hd1/zkf/RailData/images/validation/6000.jpg"
    input_img = Image.open(img_path).convert("RGB")

    # 目标类名称
    cls_name = "pantograph"
    text = f"a photograph of {cls_name}"
    
    # 使用BLIP进行文本-图像输入处理
    inputs = processor(input_img, text,return_tensors="pt").to("cuda")  

    # 使用BLIP生成新的文本描述，进行增强
    out = model.generate(**inputs)
    texts = processor.decode(out[0], skip_special_tokens=True)
    texts = text + texts[len(text):]       # 增强目标类别描述，加入更多语义信息

    print("**** blip_prompt: "+texts+"****")    # 打印生成的增强文本