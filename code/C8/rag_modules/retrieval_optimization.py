"""
检索优化模块
核心职责：混合检索 = 向量检索 + BM25 关键词检索，用 RRF 算法融合两路结果

"""

import logging
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: 已构建的 FAISS 向量数据库
            chunks: 所有子块文档（BM25 需要在内存中建倒排索引）
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()  # 初始化时立即设置两个检索器

    def setup_retrievers(self):
        """设置向量检索器和 BM25 检索器"""
        logger.info("正在设置检索器...")

        # 向量检索器：基于语义相似度
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",  # 使用余弦相似度搜索
            search_kwargs={"k": 5}  # 每次返回最相似的 5 个文档
        )

        # BM25 检索器：基于关键词匹配
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5  # 每次返回评分最高的 5 个文档
        )

        logger.info("检索器设置完成")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和 BM25 检索，使用 RRF 重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和 BM25 检索结果
        vector_docs = self.vector_retriever.invoke(query)  # 语义检索
        bm25_docs = self.bm25_retriever.invoke(query)  # 关键词检索

        # 使用 RRF 重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        - 先召回更多候选（top_k * 3），再用 metadata 过滤，保证过滤后仍有足够结果
        filters 示例：
        {"category": "素菜"} → 只返回素菜类文档
        {"difficulty": "简单"} → 只返回简单难度文档
        {"category": ["素菜", "汤品"]} → 支持列表，表示"或"关系

        Args:
            query: 查询文本
            filters: 元数据过滤条件
            top_k: 返回结果数量
            
        Returns:
            过滤后的文档列表
        """
        # 多召回 3 倍，再过滤，避免过滤后结果太少
        docs = self.hybrid_search(query, top_k * 3)

        # 应用元数据过滤
        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        # 列表过滤：metadata 值必须在列表中
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        # 精确匹配：metadata 值必须等于过滤值
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    # 文档没有这个 metadata 字段，视为不匹配
                    match = False
                    break

            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 排名融合算法

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}  # {文档哈希 ID: 累计 RRF 分数}
        doc_objects = {}  # {文档哈希 ID: Document 对象} 用于最后构建结果列表

        # 计算向量检索结果的 RRF 分数
        for rank, doc in enumerate(vector_docs):
            # 用 page_content 的哈希值作为文档唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank + 1}: RRF分数 = {rrf_score:.4f}")

        # 计算 BM25 检索结果的 RRF 分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25 检索 - 文档{rank + 1}: RRF 分数 = {rrf_score:.4f}")

        # 按最终分数降序排列
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将 RRF 分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score  # 把分数写入 metadata，方便调试
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs
