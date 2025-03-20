### CLIPer的实验代码，仅上传了做了更改的代码，其余部分与源码保持一致。

app.py：对53-56行进行修改
modified_clip/model.py：增加CLIPer_BLIP类
ovs/pipeline.py：对26-29行进行修改，对refinement进行大量修改，增加BLIP文本描述增强，增加注意力图可视化保存的功能
