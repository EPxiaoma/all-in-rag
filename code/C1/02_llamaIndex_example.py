import os

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# 加载 .env 文件中的环境变量
load_dotenv()

# 国内网络可取消注释，使用 HuggingFace 镜像站
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 中文嵌入模型（BAAI/bge-small-zh-v1.5，适合中文语义检索）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 初始化 LLM
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com",
    is_chat_model=True
)

# 3. 加载本地 markdown 文件
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 4. 构建向量索引（内部自动完成：文本分块 → embedding → 存入内存向量库）
index = VectorStoreIndex.from_documents(docs)

# 5. 创建查询引擎（封装了「检索 + 注入上下文 + 调用 LLM」的完整 RAG 流程）
query_engine = index.as_query_engine()

# 6. 获取查询引擎的提示词（用于调试和自定义 prompt）
print(query_engine.get_prompts())
print("---------------------------------------------------------------------------------------------")

# 7. 检索并生成答案
print(query_engine.query("文中举了哪些例子?"))