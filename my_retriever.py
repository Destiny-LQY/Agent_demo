import json
import numpy as np
from scipy import spatial
from typing import Dict, List, Set
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# 定义树节点类
class TreeNode:
    def __init__(self, title, content, summary, index, children=None):
        self.title = title
        self.content = content
        self.summary = summary
        self.id = index
        self.children = children if children is not None else []


# 从 JSON 数据构建树结构
def build_tree(data):
    return TreeNode(
        data["title"],
        data["content"],
        data["summary"],
        data["index"],
        [build_tree(child) for child in data["children"]]
    )


# 读取 JSON 文件
input_file_data = '/storage_server/disk5/linqiuyin/Project_Agent/RAG_Vision2/test_data_tree1.json'
with open(input_file_data, 'r', encoding='utf-8') as file:
    tree_data = json.load(file)


# 构建树
root_node = build_tree(tree_data)


# 一段文本向量化
def create_embedding(text):
    # BGE文本嵌入
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name='/storage_server/disk5/linqiuyin/Project_Agent/RAG_Vision2/bge-base-zh',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True})

    return embedding_model.embed_query(text)


# 深度优先遍历各个节点并嵌入
def dfs_traverse(node, embedding_list):
    embedding = create_embedding(node.summary)
    embedding_list.append(embedding)

    for child in node.children:
        embedding_list = dfs_traverse(child, embedding_list)

    return embedding_list


# 计算query_embedding与embeddings之间的相似度，返回 distances数组
def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


# 找到当前节点的所有子节点和子节点的子节点，返回一个字符串
def find_children(root):
    result = []
    if root is not None:
        result.append(root.title + root.content + "\n")
        for child in root.children:
            result.extend(find_children(child))

    return ''.join(result)


# 找到该目标索引的节点和进行子节点查询，返回一个字符串
def find_children_by_index(root, target_index):
    # 如果当前节点为空或找到匹配的索引，则直接返回
    if root is None or root.id == target_index:
        return find_children(root) if root.id == target_index else None

    # 遍历所有子节点
    for child in root.children:
        content = find_children_by_index(child, target_index)
        if content is not None:
            return content

    # 如果没有找到匹配的索引，则返回 None
    return None


# 找到该目标索引的节点，返回一个字符串
def find_content_by_index(root, target_index):
    # 如果当前节点为空或找到匹配的索引，则直接返回
    if root is None or root.id == target_index:
        return root.content if root.id == target_index else None

    # 遍历所有子节点
    for child in root.children:
        content = find_content_by_index(child, target_index)
        if content is not None:
            return content

    # 如果没有找到匹配的索引，则返回 None
    return None


def retrieve_tree(query, node, top_k=1, find=True):
    """
    Args:
        query (str): 要查询的内容
        node (TreeNode): 要查询的节点
        top_k (int): 选取前几个，默认为1
        find(bool):查询模式，是否查找子节点，默认为查找

    Returns:
        results(str)：查询结果
    """

    # 参数验证
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串类型")

    if not isinstance(node, TreeNode):  # 假设 Node 是某种定义好的节点类
        print(type(node))
        print(type(TreeNode))
        raise ValueError("node 必须是 Node 类型")

    query_embedding = create_embedding(query)
    # print(query_embedding)
    node_embeddings = dfs_traverse(node, [])

    distances = distances_from_embeddings(query_embedding, node_embeddings, distance_metric="cosine")
    # print(distances)

    indices = np.argsort(distances)[0:top_k]

    results = ''
    # 根据索引找到对应节点及其子节点的文本内容
    if find:
        for index in indices:
            results += find_children_by_index(node, index)
    else:
        for index in indices:
            results += find_content_by_index(node, index)
    return results


query = "关于PSL2钢管的硬度试验应该怎么做？"
retrieve_results = retrieve_tree(query, root_node, top_k=1)
print(retrieve_results)
# print(type(retrieve_results))
