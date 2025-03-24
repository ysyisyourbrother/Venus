# https://huggingface.co/BAAI/BGE-VL-base/tree/main
import torch
import sys
sys.path.append('..')
from utils.rag import load_embedding_model
from utils.vedio import process_video
import json
import os
import argparse
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument("--model_path", default="/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base" )
    parser.add_argument("--video_path",  default="../demo/lake.mp4")
    return parser.parse_args()
args = parse_args()
model = load_embedding_model(args.model_path ) # or "BAAI/BGE-VL-large"
# print(model)
with torch.no_grad():
    query = model.encode(
        text = "Cute kitty"
    )
    print(query.shape)
    candidates = model.encode(
        images = ["./bus.jpg",  "./cat.jpg"],
        text = ["", "Cute kitty"]
    )
    print(candidates.shape)
    scores = query @ candidates.T
print(scores)



# load MME data

data_path = "/root/nfs/download/dataset/lmms-lab/Video-MME" # mp4 data path
with open("../eval/short.json", 'r', encoding='utf-8') as file:
    mme_data = json.load(file)
# 选择一个视频
index =3 
item = mme_data[index]
print("video id:",item['video_id'])
video_path = os.path.join(data_path, item['url'] + ".mp4")
print("video path:", video_path)
video_path =  args.video_path  # your video path
max_frames_num = 5
frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True) #  
print(frames.shape) # (max_frames_num, 360, 640, 3)
vector =  model.encode(images = frames) # 
print(vector.shape) # torch.Size([max_frames_num, 512])



question = item['questions'][0]
query = [question['question']]
for o in question['options']:
    query.append(o)
print(query)
with torch.no_grad():
    vector =  model.encode(text = query) # 
    print(vector.shape)  # torch.Size([max_frames_num, 512])
print("=====")
print("test bge 批处理文本")
with torch.no_grad():
    vector =  model.encode(
    text = ["Cute", "Cute kitty","dog", "animal", "cat", "kitty"]
    )
    print(vector.shape)
    print(vector[0][0:10])
with torch.no_grad():
    vector =  model.encode(
    text = ["Cute" ]
    )
    print(vector.shape)
    print(vector[0][0:10])
 


# test bge 批处理
print("=====")
print("test bge 批处理 frame")
raw_frames = [f for f in frames]

with torch.no_grad():
    vector =  model.encode(
    images = raw_frames,
    )
    print( "vector",vector.shape)
    print(vector[0][0:10])
with torch.no_grad():
    vector =  model.encode(
    images = raw_frames[:1],
    )
    
    print( "vector",vector.shape)
    print(vector[0][0:10])
print("=====")
print("test bge 批处理 frame + text")
with torch.no_grad():
    vector =  model.encode(
    images = raw_frames,
    text = ["Cute", "Cute kitty", "Cute kitty", "Cute kitty", "Cute kitty"]
    )
    print( "vector",vector.shape)
    print(vector[0][0:10])
with torch.no_grad():
    vector =  model.encode(
    images = raw_frames[:1],
    text = ["Cute" ]
    )
    print( "vector",vector.shape)
    print(vector[0][0:10])
    
print("=====")

with torch.no_grad():
    print("frame + 空字符串")
    vector =  model.encode(
    images = raw_frames[:1],
    text = ["[PAD]" ]
    )
    print( "vector",vector.shape)
    print(vector[0][0:10])
with torch.no_grad():
    print("frame only")
    vector =  model.encode(
    images = raw_frames[:1],
    )
    print( "vector",vector.shape)
    print(vector[0][0:10])