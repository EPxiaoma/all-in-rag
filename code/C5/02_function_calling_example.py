import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# 初始化 LLM
client = init_chat_model(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 1. 定义工具 Schema（LangChain 支持直接传 OpenAI 格式的 dict）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

# 将工具绑定到模型，返回一个新的可调用对象
client_with_tools = client.bind_tools(tools)

# 2. 第一轮：用户提问，模型决策是否调用工具
messages = [HumanMessage(content="杭州今天天气怎么样？")]
print(f"User> {messages[0].content}\n")  # ✅ LangChain 消息对象用 .content 属性

message = client_with_tools.invoke(messages)

# 3. 执行工具，并将结果返回模型
if message.tool_calls:
    print("--- 模型发起了工具调用 ---")
    tool_call = message.tool_calls[0]
    print(f"工具名称: {tool_call['name']}")
    print(f"工具参数: {tool_call['args']}")

    messages.append(message)  # 将模型回复加入历史

    # 模拟执行工具
    tool_output = "24℃，晴朗"
    print(f"--- 执行工具并返回结果 ---")
    print(f"工具执行结果: {tool_output}\n")

    messages.append(ToolMessage(
        content=tool_output,
        tool_call_id=tool_call['id']
    ))

    # 4. 第二轮：将工具结果返回模型，获取最终答案
    print("--- 将工具结果返回给模型，获取最终答案 ---")
    final_message = client_with_tools.invoke(messages)
    print(f"Model> {final_message.content}")
else:
    print(f"Model> {message.content}")