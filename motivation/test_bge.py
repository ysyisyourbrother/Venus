# https://huggingface.co/BAAI/BGE-VL-base/tree/main
import torch
import sys
sys.path.append('..')
from utils.rag import load_embedding_model
from utils.vedio import process_video

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
MODEL_NAME = args.model_path # or "BAAI/BGE-VL-large"
model = load_embedding_model(MODEL_NAME)
print(model)
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
# load video and test 
video_path =  args.video_path  # your video path
max_frames_num =   2
frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True) #  
print(frames.shape) # (2, 360, 640, 3)
vector =  model.encode(images = frames) # 
print(vector.shape) # torch.Size([2, 512])