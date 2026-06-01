import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 .env 文件中的环境变量
load_dotenv()

# 国内网络可取消注释，使用 HuggingFace 镜像站
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 加载本地 markdown 文件
markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 2. 文本分块（默认按换行/标点递归切分，保留语义完整性）
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 3. 中文嵌入模型（BAAI/bge-small-zh-v1.5，适合中文语义检索）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 4. 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 5. 检索增强生成
question = "文中举了哪些例子？"

# 向量相似度检索，取最相关的 top-3 片段
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 6. 初始化 LLM
prompt = ChatPromptTemplate.from_template(
    """
    请根据下面提供的上下文信息来回答问题。
    请确保你的回答完全基于这些上下文。
    如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

    上下文:
    {context}

    问题: {question}

    回答:
    """
)

llm = init_chat_model(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 将检索结果注入提示词，调用 LLM 生成答案
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)