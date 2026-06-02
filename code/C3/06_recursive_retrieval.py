import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.deepseek import DeepSeek

load_dotenv()

# 1. 配置模型
# 配置全局 LLM
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
# 配置全局向量模型
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 2.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'
# 打开 Excel 文件，获取 ExcelFile 对象（不立即读取数据，便于按 sheet 分批加载）
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    # 读取当前工作表的数据为 DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine 引擎
    # 该引擎能够将自然语言问题转换为 Pandas 代码并执行，返回查询结果
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)

    # 从工作表名称（格式如"年份_1994"）中提取年份数字
    year = sheet_name.replace('年份_', '')
    # 为该工作表生成一段自然语言摘要，作为顶层索引的检索依据
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"

    # 创建 IndexNode（索引节点）：
    #   - text：摘要文本，用于向量化后与用户查询进行相似度匹配
    #   - index_id：与 df_query_engines 的 key 对应，供递归检索器定位子引擎
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)

    # 保存 sheet 名称到其查询引擎的映射，供后续递归检索器调用
    df_query_engines[sheet_name] = query_engine

# 3. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 4. 创建递归检索器
# 4.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 4.2 创建递归检索器
#     工作流程：
#       ① 用 vector_retriever 在摘要节点中检索，找到最匹配的 IndexNode
#       ② 根据该 IndexNode 的 index_id 在 query_engine_dict 中找到对应的
#          PandasQueryEngine（子引擎）
#       ③ 将原始问题转发给子引擎，由其对具体 DataFrame 执行精确查询
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 5. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 6. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
response = query_engine.query(query)
print(f"回答: {response}")