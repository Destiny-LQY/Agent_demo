from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from Retriver_Tool_2 import rendered_tools

system_prompt = f"""\
您是一个助手，可以访问以下一组检索工具。
以下是每个工具的名称和描述：
{rendered_tools}
根据用户输入，你需要思考选择使用哪个工具进行检索,返回要使用的工具的名称和输入。
将您的响应作为具有'name'和'arguments'键的JSON块返回。
'arguments'应该是一个字典，其中键对应于参数名称，值对应于请求的值。
期望的请求值是包含引用文本的语句.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

"""___________________________________________________________________________________------"""
extract_example = [
    {
        "context": """
        ### A.7.5 无损检测
         无损检测宜按照 API Spec 5L:2018 中附录K进行。
         """,
        "reference": "API Spec 5L:2018 中附录K"
    },
    {
        "context": """
        ### A.7.6 HIC复验
        如果一组 HIC试验试样不符合验收极限，购方和制造商宜协商确定复验要求。如适用，宜按
        API Spec 5L:2018 中10.2.11的规定重新处理。
         """,
        "reference": "API Spec 5L:2018 中10.2.11"
    },
    {
        "context": """
        A.7.4.2.2 无缝管的硬度试验位置见 API Spec 5L:2018 中图 H.1 a)所示，但下列情况除外：
        a)对于t<4.0 mm的钢管，仅宜在厚度中部的横向进行试验；
        b) 对于4.0 mm≤t<6.0 mm的钢管，仅宜在内表面和外表面的横向进行试验；
        c)如果协议，每一全厚度位置的三点(见 API Spec 5L:2018 中图 H.1 a)所示]压痕硬度试验是可接受的。
        """,
        "reference": "API Spec 5L:2018 中图 H.1 a"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{context}"),
        ("ai", "{reference}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=extract_example,
)
# print(few_shot_prompt.invoke({}).to_messages())
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个信息提取助手，如果上下文片段内引用了其他文献的内容，能从中提取被引用的文献名称。"),
        few_shot_prompt,
        ("human", "{context}"),
    ]
)
"""---------------------------------------------------------------------------------------"""

# if __name__ == "__main__":
#     print(final_prompt)
