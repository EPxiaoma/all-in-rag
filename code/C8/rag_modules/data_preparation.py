"""
数据准备模块
核心职责：加载食谱 Markdown 文件 → 打元数据标签 → 按标题切分子块 → 维护父子映射
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""

    # 文件夹名（英文）→ 展示标签（中文）的映射
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    # 所有中文分类标签的列表（去重），供过滤时使用
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    # 所有难度标签，按星级对应
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']

    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        
        Args:
            data_path: 存放 .md 食谱文件的根目录路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []  # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射：这个映射表的作用：检索到子块后，能快速找到它属于哪个完整食谱文档

    def load_documents(self) -> List[Document]:
        """
        遍历数据目录，读取所有 .md 文件，生成 Document 对象列表
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")

        documents = []
        data_path_obj = Path(self.data_path)

        # rglob("*.md") 递归搜索所有子目录中的 .md 文件
        for md_file in data_path_obj.rglob("*.md"):
            try:
                # 直接读取原始 Markdown 内容（保留 #、## 等标题符号，后续分割器依赖这些）
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()  # MD5 哈希：把字符串路径变成 32 位十六进制字符串作为 ID

                # 创建 LangChain Document 对象
                doc = Document(
                    page_content=content,  # 文档正文
                    metadata={
                        "source": str(md_file),  # 文件完整路径（调试用）
                        "parent_id": parent_id,  # 父文档唯一 ID
                        "doc_type": "parent"  # 标记这是父文档（区别于切分后的子块）
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")

        # 给每个文档补充分类、菜名、难度等元数据
        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据

        从文件路径和文件内容中自动提取元数据，直接修改 doc.metadata 字典
        提取三类信息：
        1. 分类（category）：从路径中的文件夹名判断
        2. 菜名（dish_name）：从文件名（去掉 .md 后缀）获取
        3. 难度（difficulty）：从正文中统计 ★ 的数量判断

        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', ''))
        path_parts = file_path.parts

        # 提取菜品分类
        doc.metadata['category'] = '其他'
        for key, value in self.CATEGORY_MAPPING.items():  # 如果路径中包含 CATEGORY_MAPPING 中的关键词
            if key in path_parts:
                doc.metadata['category'] = value
                break

        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem  # file_path.stem 就是不带后缀的文件名，例如 '红烧肉'

        # 分析难度等级
        # 通过统计文档中的星星数量判断难度
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """类方法：对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """类方法：对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        """
        对已加载的父文档进行结构化切分，生成子块列表

        切分策略：按 Markdown 标题层级（#、##、###）切分
        每个子块会继承父文档的 metadata，并添加自己的 chunk_id、位置索引等信息

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行 Markdown 结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用 Markdown 标题分割器
        chunks = self._markdown_header_split()

        # 为每个 chunk 的元数据补充批次索引和大小信息
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id'] = str(uuid.uuid4())  # 如果没有 chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['batch_index'] = i  # 在全局 chunks 列表中的位置
                chunk.metadata['chunk_size'] = len(chunk.page_content)  # 字符数，统计用

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用 MarkdownHeaderTextSplitter 按标题层级切分

        切分后每个 chunk：
        - page_content：该标题下的正文内容
        - metadata：从父文档继承 + 新增 chunk_id、parent_id、doc_type、chunk_index

        【父子检索原理】
        父文档 = 完整食谱（用于最终喂给 LLM）
        子块   = 食谱某一节（用于向量检索，粒度小，匹配精准）
        检索时用子块命中，然后通过 parent_child_map 回溯到完整父文档

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 一级标题，通常是菜品名称
            ("##", "二级标题"),   # 二级标题，如"必备原料"、"制作步骤"
            ("###", "三级标题")   # 三级标题，如"简易版本"、"进阶版本"
        ]

        # 创建 Markdown 分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题文本在 page_content 中，方便 LLM 理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 检查文档内容是否包含 Markdown 标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))
                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对该父文档进行 Markdown 分割
                md_chunks = markdown_splitter.split_text(doc.page_content)
                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())     # 每个子块分配一个随机唯一 ID

                    # 先把父文档的所有 metadata 复制到子块
                    chunk.metadata.update(doc.metadata)
                    # 再添加子块特有的字段
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i  # 在父文档中的位置
                    })

                    # 在全局映射表中记录 子块ID → 父文档ID
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 分割失败时，把整个文档当作一个 chunk 直接加入
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档

        Args:
            category: 菜品分类

        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档

        Args:
            difficulty: 难度等级

        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(
                self.chunks) if self.chunks else 0
        }

    def export_metadata(self, output_path: str):
        """
        导出元数据到 JSON 文件

        Args:
            output_path: 输出文件路径
        """
        import json

        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据检索命中的子块，回溯找到对应的完整父文档

        - 同一父文档被多个子块命中时，只返回一次（去重）
        - 命中子块越多的父文档，相关性越高，排在前面

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        parent_relevance = {}   # {父文档ID: 被命中的次数}
        parent_docs_map = {}    # {父文档ID: Document对象} 缓存，避免重复遍历

        # 收集所有相关的父文档 ID 和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 累计该父文档被命中的次数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 如果还没缓存过这个父文档，去 self.documents 里找
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按命中次数降序排列父文档 ID
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                   key=lambda x: parent_relevance[x],
                                   reverse=True)

        # 按排序顺序构建最终父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
