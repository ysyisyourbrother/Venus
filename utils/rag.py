from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
from .vision_encoder import process_images
from ultralytics import YOLO
import easyocr
tokenizer = AutoTokenizer.from_pretrained('/root/nfs/download/facebook/contriever')
model = AutoModel.from_pretrained('/root/nfs/download/facebook/contriever')
def text_to_vector(text, max_length=512):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs.last_hidden_state.shape) # torch.Size([1, 10, 768])
    # print(outputs.last_hidden_state.mean(dim=1).shape) # torch.Size([1, 768])
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy() 
def load_embedding_model(model_path,device = "cuda"):
    if 'BGE' in model_path:
        print("load BGE model:",model_path)
        from .modeling_MMRet_CLIP import CLIPModel
        model = CLIPModel.from_pretrained(model_path).to(device)
        model.set_processor(model_path)
        model.eval()
        return model
    else:
        raise NotImplementedError
def get_ocr_text(ocr_model,frames):
    # copy from https://github.com/Leon1207/Video-RAG-master/blob/main/vidrag_pipeline/vidrag_pipeline.py#L96
    text_set = []
    # 多个帧的结果
    ocr_docs = []
    for img in frames:
        ocr_results = ocr_model.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 : # and text not in text_set: ? 
                det_info += f"{text}; "
                text_set.append(text)
        ocr_docs.append(det_info)
    return ocr_docs
def get_det_text(yolo_model,frames):
    
    from collections import Counter
    threshold = 0.5 #TODO:
    results_list = [] 
    batch = 32 #TODO:
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            batch_frames = frames[i:i+batch]
            results = yolo_model(batch_frames,stream=True,batch = len(batch_frames))
            results_list.extend(results)
    det_docs = []
    for result in results_list:
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        confs = [c.item() for c in confs]
        filtered_names = [name for name, conf in zip(names, confs) if conf >= threshold]
        if len(filtered_names) == 0:
            det_docs.append( "")
        else:
            object_counts = Counter(filtered_names)
            descriptions = ["{} {}".format(count, name if count == 1 else name + "s") for name, count in object_counts.items()]
            final_description = "The image contains " + ", ".join(descriptions) + "."
            det_docs.append(final_description)
    return det_docs


def get_image_vector_with_text(frames,embedding_model,yolo_model,ocr_model):
    det_docs = get_det_text(yolo_model,frames)
    ocr_docs = get_ocr_text(ocr_model,frames)
    vectors = []
    with torch.no_grad():
        for i in range(len(frames)):
            
            frame = frames[i]
            ocr_doc = ocr_docs[i]
            det_doc = det_docs[i]
            text = ""
            if len(ocr_doc) > 0:
                text += "\nVideo OCR information: "+ "; ".join(ocr_docs)
            if len(det_doc) > 0:
                text += "\nVideo DET information: "+ "; ".join(det_docs)
            if len(text) != 0:
                # print(type(frame))
                # print( type(text) )
                # print("text",text)
                print("image and text")
                images = embedding_model.processor(images=frame, return_tensors="pt") ["pixel_values"].to(embedding_model.device)
                text = embedding_model.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(embedding_model.device)
                vector =embedding_model.encode_multimodal(images = images, text = text)
            else:
                print("image only")
                vector = embedding_model.encode(images = frame)
            # 实际上, 把single key frame + text info 加入到索引中
            if vector.is_cuda:
                vector = vector.cpu()
            vectors.append(vector)
        return np.vstack(vectors) , ocr_docs, det_docs
    
class VideoDataBase:
    def __init__(self, embedding_model, dimension, yolo_model, ocr_model ):
        self.index = None
        self.frame_list = []
        self.ocr_list = []
        self.det_list = []
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) #    这里必须传入一个向量的维度，创建一个空的索引
        self.yolo_model = yolo_model
        self.ocr_model = ocr_model
    def get_query_vector(self,query):
        with torch.no_grad():
            # print("query",query)
            query = self.embedding_model.encode(text = query)
            #TODO: 是否对齐
            # print("query shape",query.shape)
            return query.cpu().numpy() 
    def add_frames(self,frames):
        if len(frames) == 0:
            return
        assert self.index is not None
        assert isinstance(frames, list)
        # 同时处理多个帧， 因为yolo 适合批处理
        vectors, ocr_docs, det_docs = get_image_vector_with_text(frames, self.embedding_model,self.yolo_model, self.ocr_model)
        print("vectors",vectors.shape)
        # print("ocr_docs",ocr_docs)
        # print("det_docs",det_docs)
        self.index.add(vectors)
        self.frame_list.extend(frames)
        self.ocr_list.extend(ocr_docs)
        self.det_list.extend(det_docs)
    def get_raw_data(self, idx):
        # 根据索引获取原始数据，以字典形式返回
        return {
        "frame": self.frame_list[idx],
        "ocr": self.ocr_list[idx],
        "det": self.det_list[idx]
    }
    def retrieve_documents_top_k(self, query,  top_k=5):        
        query_vector = self.get_query_vector(query)
        D, I = self.index.search( query_vector, top_k)  # I.shape = (1, top_k)
        idx = I[0].tolist() 
        top_documents = [self.get_raw_data(i) for i in idx if i != -1]
        return top_documents,idx

    def retrieve_documents_with_dynamic(self, query, threshold=0.4):
        # query转化为向量
        query_vector = self.get_query_vector(query)
        lims, D, I = self.index.range_search(query_vector, threshold)
        start = lims[0]
        end = lims[1]
        I = I[start:end]
        if len(I) == 0:
            top_documents = []
            idx = []
        else:
            idx = I.tolist()
            top_documents = [self.get_raw_data(i) for i in idx  if i != -1]
        return top_documents, idx
    def print_index_ntotal(self):
        print(self.index.ntotal)
class TextDataBase:
    def __init__(self, embedding_model, dimension):
        self.index = None
        self.documents = []
        self.index = faiss.IndexFlatIP(dimension) #    这里必须传入一个向量的维度，创建一个空的索引
        self.embedding_model = embedding_model
    def encode_text(self, text):
        # print(text)
        # print(len(text))
        with torch.no_grad():
            text = self.embedding_model.encode(text = text)
            return text.cpu().numpy()
    def add_documents(self,documents):
        if len(documents) == 0:
            return
        assert self.index is not None
        assert isinstance(documents, list)
 
        document_vectors =  [ self.encode_text(doc) for doc in documents]
        document_vectors = np.vstack(document_vectors)
        print("document_vectors",document_vectors.shape)
        self.index.add(document_vectors)
        self.documents.extend(documents)
    def retrieve_documents_top_k(self, query,  top_k=5):        
        query_vector =self.encode_text(query)
        print("query_vector",query_vector.shape)
        D, I = self.index.search( query_vector, top_k)  # I.shape = (1, top_k)
        idx = I[0].tolist() 
        top_documents = [ self.documents[i] for i in idx if i != -1]
        return top_documents,idx
    def retrieve_documents_with_dynamic(self, query, threshold=0.4):
        # query转化为向量
        query_vector = self.encode_text(query)
        lims, D, I = self.index.range_search(query_vector, threshold)
        start = lims[0]
        end = lims[1]
        I = I[start:end]
        if len(I) == 0:
            top_documents = []
            idx = []
        else:
            idx = I.tolist()
            top_documents = [ self.documents[i] for i in idx if i != -1]
        return top_documents, idx
    def print_index_ntotal(self):
        print(self.index.ntotal)
class DataBase:
    def __init__(self):
        self.index = None
        self.documents = []
    def get_query_vector(self,queries):
        
        if isinstance(queries, list):
            query_vectors = np.array([text_to_vector(query) for query in queries])
            average_query_vector = np.mean(query_vectors, axis=0)
            query_vector = average_query_vector / np.linalg.norm(average_query_vector)
            query_vector = query_vector.reshape(1, -1)
        else:
            query_vector = text_to_vector(queries)
            query_vector = query_vector / np.linalg.norm(query_vector)
            query_vector = query_vector.reshape(1, -1)
        return query_vector
    def retrieve_documents_top_k(self, queries,  top_k=5):        
        query_vector = self.get_query_vector(queries)
        D, I = self.index.search( query_vector, top_k)  # I.shape = (1, top_k)
        idx = I[0].tolist() 
        top_documents = [self.documents[i] for i in idx if i != -1]
        return top_documents,idx
            
    def retrieve_documents_with_dynamic(self, queries, threshold=0.4):
        # query转化为向量
        query_vector = self.get_query_vector(queries)
        # 使用index.range_search(query_vector, threshold)进行范围搜索，返回符合相似度阈值的文档的索引I和距离D。该方法会返回两个结果：
        lims, D, I = self.index.range_search(query_vector, threshold)
        start = lims[0]
        end = lims[1]
        I = I[start:end]
        if len(I) == 0:
            top_documents = []
            idx = []
        else:
            idx = I.tolist()
            top_documents = [self.documents[i] for i in idx]
        return top_documents, idx
 
    def index_data_add(self,documents ):
        document_vectors = np.array([text_to_vector(doc) for doc in documents])
        print("document_vectors",document_vectors.shape) # (3, 768)
        if self.index is None:
            # document_vectors = document_vectors / np.linalg.norm(document_vectors, axis=1, keepdims=True)
            dimension = document_vectors.shape[1]
            print("document_vectors",document_vectors.shape  )
            print("dimension",dimension  )
            self.index = faiss.IndexFlatIP(dimension) #    这里必须传入一个向量的维度，创建一个空的索引
        self.index.add(document_vectors)
        self.documents.extend(documents)
        
    def print_index_ntotal(self):
        print(self.index.ntotal)
 
 
