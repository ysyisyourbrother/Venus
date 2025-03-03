from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

tokenizer = AutoTokenizer.from_pretrained('/root/nfs/download/facebook/contriever')
model = AutoModel.from_pretrained('/root/nfs/download/facebook/contriever')

def text_to_vector(text, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_documents_with_dynamic(documents, queries, threshold=0.4):
    # query转化为向量
    print("query",queries)
    if isinstance(queries, list):
        query_vectors = np.array([text_to_vector(query) for query in queries])
        average_query_vector = np.mean(query_vectors, axis=0)
        query_vector = average_query_vector / np.linalg.norm(average_query_vector)
        query_vector = query_vector.reshape(1, -1)
    else:
        query_vector = text_to_vector(queries)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1)
    # 每个文档向量进行归一化处理
    document_vectors = np.array([text_to_vector(doc) for doc in documents])
    document_vectors = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
    dimension = document_vectors.shape[1]
    
    index = faiss.IndexFlatIP(dimension) #    这里必须传入一个向量的维度，创建一个空的索引
    index.add(document_vectors) # 添加文档向量到索引中
    # 使用index.range_search(query_vector, threshold)进行范围搜索，返回符合相似度阈值的文档的索引I和距离D。该方法会返回两个结果：
    lims, D, I = index.range_search(query_vector, threshold)
    start = lims[0]
    end = lims[1]
    I = I[start:end]

    if len(I) == 0:
        top_documents = []
        idx = []
    else:
        idx = I.tolist()
        top_documents = [documents[i] for i in idx]

    return top_documents, idx
