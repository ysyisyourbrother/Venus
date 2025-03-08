import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import socket

import pickle
import os
 
def save_frames(frames):
    file_paths = []
    save_dir = "restore"  # 存储目录
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = os.path.abspath(os.path.join(save_dir, f'frame_{i}.png'))  # 获取绝对路径
        img.save(file_path)
        file_paths.append(file_path)

    print(file_paths)  # 打印绝对路径
    return file_paths  # 返回绝对路径列表


    
def get_det_docs(frames, prompt):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except:
            res = []
    return res

images = ["cat.jpg", "bus.jpg"]
frames = [ np.asarray(Image.open(image) ) for image in images]
 
device = "cuda"
# clip_model = CLIPModel.from_pretrained("/root/nfs/codespace/llm-models/MLLM/openai/clip-vit-large-patch14-336",)
# clip_processor = CLIPProcessor.from_pretrained("/root/nfs/codespace/llm-models/MLLM/openai/clip-vit-large-patch14-336")
# clip_model.to(device)
# tensor =  clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
# print(tensor.shape)
 
 
det_docs =  get_det_docs( frames ,  "cat")  # 

print(det_docs)
# det_docs = det_preprocess(det_docs, location=L, relation=R, number=N) 