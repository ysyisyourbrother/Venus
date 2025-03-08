from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import sys
sys.path.append('..')
from utils.rag import DataBase
docs = ["I like Xiaoding's articles", "I hate Xiaoding's creative content", "I really like the articles written by Xiaoding"]
query = "I love reading Xiaoding's articles"
db = DataBase()
db.index_data_add(docs)
db.print_index_ntotal()
top_documents, idx = db.retrieve_documents_with_dynamic(query)
print("top_documents",top_documents)
print("idx",idx)

# add documents
documents = ["NO! I like tennis!","Let me know if you need further modifications or explanations!"]
db.index_data_add(documents)
db.print_index_ntotal()
top_documents, idx = db.retrieve_documents_with_dynamic(query)
print("top_documents",top_documents)
print("idx",idx)


print("top k result")
db.print_index_ntotal()
top_documents, idx = db.retrieve_documents_top_k(query , top_k=2)
print("top_documents",top_documents)
print("idx",idx)