### 🔧 CLIPer 实验代码说明

**注：本项目仅包含核心修改代码，未改动部分与原始代码库保持完全一致**

#### 📝 关键代码变更
**`app.py`**
- 53-56行代码进行修改

**`modified_clip/model.py`**
- 新增 `CLIPer_BLIP` 融合类

**`ovs/pipeline.py`** 
- 重构26-29行预处理流程
- 升级refinement模块架构
- 集成 **BLIP文本描述增强器** 
- 自动保存注意力热力图
