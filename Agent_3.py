from Prompt_2 import prompt
from dotenv import load_dotenv
load_dotenv()
import json
from langchain_community.chat_models import QianfanChatEndpoint
llm = QianfanChatEndpoint(model_name="ERNIE-3.5-turbo")

from langchain_core.output_parsers import JsonOutputParser
from Retriver_Tool_2 import invoke_tool
# chain = prompt | llm | JsonOutputParser() | invoke_tool
from langchain_core.runnables import RunnablePassthrough
chain = prompt | llm | JsonOutputParser() | RunnablePassthrough.assign(output=invoke_tool)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

query = "如何对钢管进行金相组织检验?"
response = invoke_tool({"name": "RAG_1", "arguments": query})
print("主文档检索结果:", response)
# from my_retriever import retrieve_results
# query = "关于PSL2钢管的硬度试验应该怎么做？"
# response = retrieve_results
if __name__ == "__main__":
    ref = chain.invoke({"input": response})
    retriever_result = ref["output"]
    context = response + retriever_result
    # context = str(response)+"\n"+str(ref["output"])
    # print(context)
    prompt_1 = SystemMessage(content="你是一个检测专家")
    new_prompt = (
            prompt_1 + HumanMessage(content=query) + AIMessage(content=context) + "{input}"
    )
    QA_chain = new_prompt | llm

    # print(ref.content)

    # print(ref)
    print("调用的工具名称：", ref["name"])
    print("调用的工具参数:", ref["arguments"])
    print("检索结果:", ref["output"])

    result = QA_chain.invoke({"input": "请梳理上下文片段，输出专业答案"})
    print(result.content)
    new_data = {
        "test_10": {
          "query": query,
          "主文档检索结果:": response,
          "调用的工具名称": ref["name"],
          "调用的工具参数:": ref["arguments"],
          "检索结果": ref["output"],
          "answer": result.content
        }
    }
    # with open("agent_response.json", "w") as f:
    #     json.dump(new_data, f, ensure_ascii=False, indent=4)

    with open("agent_response.json", "r") as f:
        data = json.load(f)
    data.update(new_data)
    with open("agent_response.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
