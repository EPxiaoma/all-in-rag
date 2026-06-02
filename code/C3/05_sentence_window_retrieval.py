import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek

# 1. 配置模型
# 配置全局 LLM
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY"))
# 配置全局向量模型
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 2. 加载文档
documents = SimpleDirectoryReader(
    input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 3. 创建节点与构建索引
# 3.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # 当前句子前后保留 3 个句子作为上下文
    window_metadata_key="window",   # 保存上下文窗口的 metadata 字段名
    original_text_metadata_key="original_text",   # 保存原始文本的 metadata 字段名
)
# 将文档切分成多个句子节点
sentence_nodes = node_parser.get_nodes_from_documents(documents)

# 创建向量索引
sentence_index = VectorStoreIndex(sentence_nodes)

# 3.2 常规分块索引（作为对照组）
base_parser = SentenceSplitter(chunk_size=512)  # 使用固定长度切分
base_nodes = base_parser.get_nodes_from_documents(documents)    # 切分文档
base_index = VectorStoreIndex(base_nodes)   # 构建普通向量索引

# 4. 构建查询引擎
# 4.1 句子窗口检索
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

# 4.2 普通检索
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 5. 执行查询并对比结果
query = "What are the concerns surrounding the AMOC?"
print(f"查询: {query}\n")

print("--- 句子窗口检索结果 ---")
window_response = sentence_query_engine.query(query)
print(f"回答: {window_response}\n")

print("--- 常规检索结果 ---")
base_response = base_query_engine.query(query)
print(f"回答: {base_response}\n")