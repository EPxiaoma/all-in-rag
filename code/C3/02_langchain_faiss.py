from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
# 将每条原始文本包装成 Document 对象
docs = [Document(page_content=t) for t in texts]
# 初始化 HuggingFace 嵌入模型，使用北京智源的 bge-small-zh-v1.5
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 使用文档列表和嵌入模型构建 FAISS 向量存储
vectorstore = FAISS.from_documents(docs, embeddings)

# 定义本地存储路径（会生成 index.faiss 和 index.pkl 两个文件）
local_faiss_path = "./faiss_index_store"
# 保存 FAISS 索引
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 从本地路径加载已保存的 FAISS 索引
# 加载时需指定相同的嵌入模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 定义查询语句
query = "张三是谁？"
# 执行相似性搜索：将 query 向量化后，在索引中找出余弦距离最近的 k 个文档
results = loaded_vectorstore.similarity_search(query, k=1)

# 输出查询结果
print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")