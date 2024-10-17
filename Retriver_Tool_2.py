from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
import re
from langchain_core.documents import Document
import uuid

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.tools import tool
from langchain_core.tools import render_text_description
from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
"""
bge_embeddings-----------------------------------------------------------------------
"""

model_name = "/storage_server/disk5/linqiuyin/Project_Langchain/Test/BAAI/bge-small-zh-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
"""
read_PDF------------------------------------------------------------
"""
def read_PDF(file_path):
    # 使用PyPDFLoader加载PDF文档
    loader = PyPDFLoader(file_path)
    # 加载并拆分PDF文档的每一页
    pages = loader.load_and_split()
    # 初始化一个空列表，用于存储每个页面的内容和元数据
    pdf_content = []
    # 遍历每一页，将内容和元数据存储到pdf_content列表中
    for page in pages:
        pdf_content.append({'content': page.page_content,
                            'metadata': page.metadata})
    # 将每个页面的内容和元数据转换为Document对象
    documents = [Document(page_content=x['content'], metadata=x['metadata']) for x in pdf_content]
    return documents

"""
split_markdown--------------------------------------------------------------------
"""
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def split_markdown_by_headers(content):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = splitter.split_text(content)
    # print("length of md_header_splits:{}".format(len(md_header_splits)))
    return md_header_splits

def split_markdown_by_sentences(documents):
    documents_splitted = []
    for document in documents:
        content = document.page_content
        sentences = re.split(r'[。！？]', content)
        documents = [Document(page_content=sentence, metadata=document.metadata) for sentence in sentences]
        documents_splitted.extend(documents)
    # print("length of md_sentence_splits:{}".format(len(documents_splitted)))
    return documents_splitted

"""
multi_vector_retriever----------------------------------------------------------------
"""
def multi_vector_retriever(content, question):
    docs = split_markdown_by_headers(content)
    #The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=bge_embeddings
    )
    # vectorstore = FAISS.from_documents(docs, bge_embeddings)

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        sentences = re.split(r'[。！？]', doc.page_content)
        _sub_docs = [Document(page_content=sentence) for sentence in sentences]
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
    # # print(len(sub_docs))
    for doc in sub_docs:
        if 0 < len(doc.page_content) < 500:
            retriever.vectorstore.add_documents([doc])

    # retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    # small_chunk_ans = retriever.vectorstore.similarity_search(question, k=1)
    # print("small_chunk_ans:{}".format(small_chunk_ans))
    # large_chunk_ans = retriever.invoke(question)[0]
    # print("large_chunk_ans:{}".format(large_chunk_ans))
    return retriever

"""
RAG_1: main_document GBT9711
"""
markdown_path = "/storage_server/disk5/linqiuyin/Project_Agent/Data/test_data.md"
@tool
def RAG_1(query: str) -> str:
    """主文档:关于石油天然气工业管线输送系统用焊管和无缝管的检测的资料。"""
    readme_content = read_markdown_file(markdown_path)
    # md_header_splits = split_markdown_by_headers(readme_content)
    # final_splits = split_markdown_by_sentences(md_header_splits)
    # documents = RecursiveCharacterTextSplitter(
    #     chunk_size=200, chunk_overlap=20
    # ).split_documents(md_header_splits)
    # 建立索引：按章节+递归字符切分
    # vector = FAISS.from_documents(documents, bge_embeddings)
    # vector.save_local("faissIndex_GBT9711")
    # 利用本地存储的索引进行检索
    # vector = FAISS.load_local("faissIndex_GBT9711", bge_embeddings, allow_dangerous_deserialization=True)
    # retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # rag_ans_list = retriever.invoke(query)
    # rag_ans = ".\n\n".join([str(ans.page_content) for ans in rag_ans_list])

    # multi-vector多向量查询：
    # small chunk（句子）—— 相似度检索
    # large chunk（章节）—— 返回上下文
    retriever = multi_vector_retriever(readme_content, query)
    rag_ans = retriever.invoke(query)[0].page_content
    return rag_ans

"""
RAG_2: reference GBT4335
"""
@tool
def RAG_2(query: str) -> str:
    """参考文档:GB/T 4335低碳钢冷轧薄板铁素体晶粒度测定法。"""
    readme_content = read_markdown_file("/storage_server/disk5/linqiuyin/Project_Agent/markdown_data/引用文本-GB_T_4335-2013/引用文本-GB_T_4335-2013.md")
    # md_header_splits = split_markdown_by_headers(readme_content)
    # final_splits = split_markdown_by_sentences(md_header_splits)
    # documents = RecursiveCharacterTextSplitter(
    #     chunk_size=200, chunk_overlap=20
    # ).split_documents(md_header_splits)
    # # 建立索引：按章节+递归字符切分
    # vector = FAISS.from_documents(documents, bge_embeddings)
    # vector.save_local("faissIndex_GBT4335")
    # 利用本地存储的索引进行检索
    # vector = FAISS.load_local("/storage_server/disk5/linqiuyin/Project_Agent/RAG_Vision1/faissIndex_GBT4335", bge_embeddings, allow_dangerous_deserialization=True)
    # retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    # rag_ans_list = retriever.invoke(query)
    # rag_ans = ".\n\n".join([str(ans.page_content) for ans in rag_ans_list])

    retriever = multi_vector_retriever(readme_content, query)
    rag_ans = retriever.invoke(query)[0].page_content
    return rag_ans
"""
RAG_3: reference GBT10561
"""
@tool
def RAG_3(query: str) -> str:
    """参考文档:GB/T 10561—2005 钢中非金属夹杂物含量的测定标准评级图显微检验法。"""
    readme_content = read_markdown_file(
        "/storage_server/disk5/linqiuyin/Project_Agent/markdown_data/引用文本-GBT10561-2005_Password_Removed/引用文本-GBT10561-2005_Password_Removed.md")
    # md_header_splits = split_markdown_by_headers(readme_content)
    # final_splits = split_markdown_by_sentences(md_header_splits)
    # documents = RecursiveCharacterTextSplitter(
    #     chunk_size=200, chunk_overlap=20
    # ).split_documents(md_header_splits)
    # # 建立索引：按章节+递归字符切分
    # vector = FAISS.from_documents(documents, bge_embeddings)
    # vector.save_local("faissIndex_GBT10561")
    # # 利用本地存储的索引进行检索
    vector = FAISS.load_local("/storage_server/disk5/linqiuyin/Project_Agent/RAG_Vision1/faissIndex_GBT10561", bge_embeddings, allow_dangerous_deserialization=True)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    rag_ans_list = retriever.invoke(query)
    rag_ans = ".\n\n".join([str(ans.page_content) for ans in rag_ans_list])

    # retriever = multi_vector_retriever(readme_content, query)
    # rag_ans = retriever.invoke(query)[0].page_content
    return rag_ans
"""
RAG_4: reference API SPEC 5L-2018
"""
@tool
def RAG_4(query: str) -> str:
    """参考文档:API SPEC 5L-2018"""
    # readme_content = read_markdown_file("/storage_server/disk5/linqiuyin/Project_Agent/markdown_data/引用文本-API/引用文本-API.md")
    # md_header_splits = split_markdown_by_headers(readme_content)
    # final_splits = split_markdown_by_sentences(md_header_splits)
    # documents = RecursiveCharacterTextSplitter(
    #     chunk_size=200, chunk_overlap=20
    # ).split_documents(final_splits)
    #
    # faissIndex= FAISS.from_documents(documents, bge_embeddings)
    # faissIndex.save_local("faissIndex_API")

    faissIndex = FAISS.load_local("/storage_server/disk5/linqiuyin/Project_Agent/RAG_Vision1/faissIndex_API", bge_embeddings, allow_dangerous_deserialization=True)
    retriever = faissIndex.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    rag_ans_list = retriever.invoke(query)
    rag_ans = ".\n\n".join([str(ans.page_content) for ans in rag_ans_list])
    return rag_ans

tools = [RAG_1, RAG_2, RAG_3, RAG_4]
# 让我们检查这些工具
# for t in tools:
#     print("--")
#     print(t.name)
#     print(t.description)
#     print(t.args)
rendered_tools = render_text_description(tools)
print(rendered_tools)
class ToolCallRequest(TypedDict):
    """一个类型化字典，显示了传递给 invoke_tool 函数的输入。"""
    name: str
    arguments: Dict[str, Any]
def invoke_tool(
    tool_call_request: ToolCallRequest, config: Optional[RunnableConfig] = None
):
    """我们可以使用的执行工具调用的函数。
    参数:
        tool_call_request: 一个包含键名和参数的字典。
            名称必须与现有工具的名称匹配。
            参数是该工具的参数。
        config: 这是 LangChain 使用的包含回调、元数据等信息的配置信息。
            请参阅有关 RunnableConfig 的 LCEL 文档。
    返回:
        请求工具的输出
    """
    try:
        tool_name_to_tool = {tool.name: tool for tool in tools}
        name = tool_call_request["name"]
        requested_tool = tool_name_to_tool[name]
        return requested_tool.invoke(tool_call_request["arguments"], config=config)
    except Exception as e:
        print(f"Error invoking tool: {e}")
        return "Error invoking tool"

if __name__ == "__main__":
    print("hello")






