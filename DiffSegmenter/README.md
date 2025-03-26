### 🏀 DiffSegmenter 实验代码说明

**注：本项目仅包含核心修改代码，未改动部分与原始代码库[DiffSegmenter](https://github.com/VCG-team/DiffSegmenter)保持完全一致**

#### 📝 关键代码变更
**`visual_code/prompt-to-prompt_stable.ipynb`**
- 增加calculate_iou、calculate_miou函数，用于计算IoU与MIoU
- 增加baseline函数，用于生成预测掩码图片
