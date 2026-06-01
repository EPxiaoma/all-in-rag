import torch

from visual_bge.visual_bge.modeling import Visualized_BGE

# 初始化模型：
# - model_name_bge: 使用 BAAI 的 bge-base-en-v1.5 作为文本编码器基座
# - model_weight: 加载包含视觉能力的预训练权重
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                       model_weight="../../models/bge/Visualized_base_en_v1.5.pth")

model.eval()    # 切换为推理模式，关闭 Dropout 等训练专用层

# 关闭梯度计算，节省显存并加速推理
with torch.no_grad():
    # 纯文本编码：将文字描述转为向量
    text_emb = model.encode(text="datawhale开源组织的logo")

    # 纯图像编码：仅用图片生成嵌入向量
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")

    # 图文融合编码：将图片 + 文字描述合并为一个多模态向量
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")

    # 另一张纯图像编码
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")

    # 另一张图文融合编码
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")