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
 
datas = ["我喜欢小丁的文章", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
datas_embedding = text_to_vector(datas)
def create_index(datas_embedding):
    # 构建索引，这里我们选用暴力检索的方法FlatL2为例，L2代表构建的index采用的相似度度量方法为L2范数，即欧氏距离
    index = faiss.IndexFlatL2(datas_embedding.shape[1])  # 这里必须传入一个向量的维度，创建一个空的索引
    index.add(datas_embedding)   # 把向量数据加入索引
    return index

def data_recall(faiss_index, query, top_k):
    query_embedding =  text_to_vector(query)
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index

datas = ["我喜欢小丁的文章", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
datas_embedding = text_to_vector(datas)
faiss_index = create_index(datas_embedding)
sim_data_Index = data_recall(faiss_index, "我爱看小丁的文章", 2)
print("相似的top2数据是：")
for index in sim_data_Index[0]:
    print(datas[int(index)] + "\n")