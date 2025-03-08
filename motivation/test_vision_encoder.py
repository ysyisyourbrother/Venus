import os
import sys
sys.path.append('..')
 
import torch.nn as nn
import torch
import math 
import argparse
from utils.vedio import process_video
import random
import time
from utils.vision_encoder import load_vision_encoder,process_images
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--vision_tower_path", default="/root/nfs/codespace/llm-models/MLLM/google/siglip-so400m-patch14-384" )
    parser.add_argument("--video_path",  default="../demo/lake.mp4")
    return parser.parse_args()
 
def test_vision_encoder(args):
    vision_encoder = load_vision_encoder(  args.vision_tower_path)
    image_processor = vision_encoder.image_processor
    video_path =  args.video_path  # your video path
    max_frames_num =   8
    frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True) #  
    
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda() 
    video = [video]
    print(len(video[0])) # 32 frames
    print( (video[0].shape)) # torch.Size([32, 3, 384, 384])

    start_time = time.perf_counter()
    iter = 10 
    for i in range(iter):
        image_features =  process_images(vision_encoder, video) 

    end_time = time.perf_counter()
    avg_iter_time = (end_time - start_time) * 1000/iter
    avg_frame_time = avg_iter_time / max_frames_num
    print("image_features",image_features.shape)
    print("image_features dtype:", image_features.dtype)
    print("avg_iter_time",avg_iter_time)
    print("avg_frame_time",avg_frame_time)
if __name__ == "__main__":
    args = parse_args()
    print(args)
    test_vision_encoder(args)