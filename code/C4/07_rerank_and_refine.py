import os
from typing import Sequence

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from transformers import AutoTokenizer, AutoModel

# 1. ColBERT 重排器定义
class ColBERTReranker(BaseDocumentCompressor):
    """
    ColBERT 风格的重排器（Reranker）。

    【核心思想】
    - 传统双塔模型：query 和 doc 各自独立编码成一个向量，点积计算相似度。
      信息损失大，因为所有语义被压缩进一个向量。
    - ColBERT（Contextualized Late Interaction over BERT）：
      query 和 doc 各自保留所有 token 的向量（token-level embeddings），
      通过 MaxSim 操作计算细粒度交互，相比单向量检索更精准。

    【MaxSim 操作】
    对 query 的每个 token，找出它与 doc 所有 token 中最相似的那个，
    然后把所有 query token 的最大相似度加总，得到最终分数。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_name = "bert-base-uncased"

        # 加载模型和分词器，这里用通用 BERT 做演示，效果有所折扣但原理相同
        object.__setattr__(self, 'tokenizer', AutoTokenizer.from_pretrained(model_name))
        object.__setattr__(self, 'model', AutoModel.from_pretrained(model_name))
        self.model.eval()
        print(f"ColBERT模型加载完成")

    def encode_text(self, texts):
        """将文本列表编码为 token 级别的向量序列"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def calculate_colbert_similarity(self, query_emb, doc_embs, query_mask, doc_masks):
        """
        ColBERT 的核心：MaxSim 相似度计算。

        参数说明：
          query_emb  : shape [1, q_len, 768]，单个 query 的 token 向量序列
          doc_embs   : shape [N, d_len, 768]，N 个文档的 token 向量序列
          query_mask : shape [1, q_len]，标记 query 哪些位置是真实 token（1）哪些是 padding（0）
          doc_masks  : shape [N, d_len]，同上，针对文档

        返回：每个文档对应的 ColBERT 分数列表，分数越高越相关。
        """
        scores = []

        for i, doc_emb in enumerate(doc_embs):
            doc_mask = doc_masks[i:i+1]

            # 计算 query 所有 token 与 doc 所有 token 的相似度矩阵
            similarity_matrix = torch.matmul(query_emb, doc_emb.unsqueeze(0).transpose(-2, -1))

            # 屏蔽 doc 的 padding 位置
            doc_mask_expanded = doc_mask.unsqueeze(1)
            similarity_matrix = similarity_matrix.masked_fill(~doc_mask_expanded.bool(), -1e9)

            # MaxSim 操作 —— 对每个 query token，取其与所有 doc token 的最大相似度
            max_sim_per_query_token = similarity_matrix.max(dim=-1)[0]

            # 屏蔽 query 的 padding 位置
            query_mask_expanded = query_mask.unsqueeze(0)
            max_sim_per_query_token = max_sim_per_query_token.masked_fill(~query_mask_expanded.bool(), 0)

            # 对所有 query token 的 MaxSim 分数求和
            colbert_score = max_sim_per_query_token.sum(dim=-1).item()
            scores.append(colbert_score)

        return scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序"""
        if len(documents) == 0:
            return documents

        # 编码查询
        query_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            query_outputs = self.model(**query_inputs)
            query_embeddings = F.normalize(query_outputs.last_hidden_state, p=2, dim=-1)

        # 编码文档
        doc_texts = [doc.page_content for doc in documents]
        doc_inputs = self.tokenizer(
            doc_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            doc_outputs = self.model(**doc_inputs)
            doc_embeddings = F.normalize(doc_outputs.last_hidden_state, p=2, dim=-1)

        # 计算ColBERT相似度
        scores = self.calculate_colbert_similarity(
            query_embeddings,
            doc_embeddings,
            query_inputs['attention_mask'],
            doc_inputs['attention_mask']
        )

        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]

        return reranked_docs



# 初始化嵌入模型
hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

# 初始化 LLM
load_dotenv()
llm = init_chat_model(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# 1. 文档加载与切分
loader = TextLoader("../../data/C4/txt/ai.txt", encoding="utf-8")
documents = loader.load()
# 递归字符分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 2. 创建向量存储和基础检索器
vectorstore = FAISS.from_documents(docs, hf_bge_embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 3. 设置ColBERT重排序器
reranker = ColBERTReranker()

# 4. 设置LLM压缩器
compressor = LLMChainExtractor.from_llm(llm)

# 5. 使用DocumentCompressorPipeline组装压缩管道
# 流程: 输入 20 个文档 → ColBERT重排 → 5 个文档 → LLM压缩 → 最终精华片段
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[reranker, compressor]
)

# 6. 用管道压缩器包装基础检索器，形成最终检索器
final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 7. 执行查询并展示结果·
query = "AI还有哪些缺陷需要克服？"
print(f"\n{'='*20} 开始执行查询 {'='*20}")
print(f"查询: {query}\n")

# 7.1 基础检索结果
print(f"--- (1) 基础检索结果 (Top 20) ---")
base_results = base_retriever.get_relevant_documents(query)
for i, doc in enumerate(base_results):
    print(f"  [{i+1}] {doc.page_content[:100]}...\n")

# 7.2 使用管道压缩器的最终结果
print(f"\n--- (2) 管道压缩后结果 (ColBERT重排 + LLM压缩) ---")
final_results = final_retriever.get_relevant_documents(query)
for i, doc in enumerate(final_results):
    print(f"  [{i+1}] {doc.page_content}\n")