from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models import QianfanChatEndpoint
llm = QianfanChatEndpoint(model_name="ERNIE-3.5-turbo")
import json
from Prompt_2 import prompt
from langchain_core.output_parsers import JsonOutputParser
from Retriver_Tool_2 import invoke_tool

from langchain_core.runnables import RunnablePassthrough
chain = prompt | llm | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)
# query = "如何对钢管进行金相组织检验?"
# query = "补焊焊接操作人园技能评定的检验"
# query = "复验用试样应如何截取?"
# query = "关于PSL2钢管的硬度试验应该怎么做？"
# query = "焊缝射线检测的设备校验"
# query = "金相组织晶粒度检验"
# response = invoke_tool({"name": "RAG_1", "arguments": query})
# print("主文档检索结果:", response)
# query = "API Spec 5L:2018中10.2.11"
'''--------------------------------------------------------------------'''
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# prompt_1 = SystemMessage(content="你是一个检测专家")
# new_prompt = (
#     prompt_1 + HumanMessage(content=query) + AIMessage(content=response) + "{input}"
# )
# new_prompt.format_messages(input="以检测专家的口吻输出答案")
# print(new_prompt)
# QA_chain = new_prompt | llm
# result = QA_chain.invoke({"input": "以检测专家的口吻输出检测步骤"})
# print(result.content)
'''-------------------------------------------------------------------'''
query = "金相组织非金属夹杂物按 GB/T 10561—2005 中方法 A进行检验的相关内容。"
# query = "方法 A"
result_RAG_3 = invoke_tool({"name": "RAG_3", "arguments": query})
print("RAG_3结果:", result_RAG_3)
'''------------------------------------------------------------------'''
# query = "复验用试样应按照 API Spec 5L:2018 中表19、API Spec 5L;2018 中表20 以及 API Spee 5L;2018中10.2.3.6的要求截取。"
# query = "晶粒度宜为GB/T4335的8级或更细，并按GB/T4335中规定进行检验。"
# response = chain.invoke({"input": query})
# print("调用的工具名词：", response["name"])
# print("问题的答案:", response["output"])
# new_data = {
#     "test_5": {
#       "query": query,
#       "tool_call": response["name"],
#       "answer": response["output"]
#     }
# }
# with open("agent_response.json", "w") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)

# with open("agent_response.json", "r") as f:
#     data = json.load(f)
# data.update(new_data)
# with open("agent_response.json", "w") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
