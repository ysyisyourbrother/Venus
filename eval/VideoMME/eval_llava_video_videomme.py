import requests
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel, WhisperForConditionalGeneration, WhisperProcessor
import copy
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm
import os
from utils.vedio import process_video
import argparse


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Video with VideoMME settings.")
    parser.add_argument(
        "--video_duration_type",
        required=True,
        type=str,
        choices=["short", "medium", "long"],
        help="Specify the video duration type: short, medium, or long."
    )
    parser.add_argument(
        "--max_frames_num",
        required=True,
        type=int,
        help="Specify the maximum number of video frames to process."
    )
    parser.add_argument(
        "--vlm_path",
        help="Path to the pre-trained weights of the LLaVA-Video-7B-Qwen2 vision-language model.",
        default="/root/nfs/codespace/llm-models/MLLM/lmms-lab/LLaVA-Video-7B-Qwen2"
    )
    parser.add_argument(
        "--mme_data_path",
        help="Path to the MME dataset",
        default= "/root/nfs/download/dataset/lmms-lab/Video-MME"
    )
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", 
                        choices=["eager", "sdpa", "flash_attention_2"])
    return parser.parse_args()
    
def llava_inference(qs, video):
    if video is not None:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n" + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    if video is not None:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=16,  
            top_p=1.0,
            num_beams=1
        )
    else:
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs 
args = parse_args() 
device = "cuda"
max_frames_num = args.max_frames_num
overwrite_config = {}
overwrite_config["mm_spatial_pool_mode"] =  "average"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    args.vlm_path , 
    None, 
    "llava_qwen", 
    torch_dtype="bfloat16", 
    load_in_8bit=False,
    load_in_4bit=False,
    overwrite_config=overwrite_config, 
    attn_implementation=args.attn_implementation,
    device_map="auto")  # Add any other thing you want to pass in llava_model_args
model.eval()
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
print(model)
data_path = args.mme_data_path # mp4 data path
print("MME data path: ", data_path)
with open(f"eval/VideoMME/{args.video_duration_type}.json", 'r', encoding='utf-8') as file:
    mme_data = json.load(file)
# save result 
os.makedirs("eval/results", exist_ok=True)
json_file = f"eval/results/eval_llava_video_videomme_{args.video_duration_type}_{max_frames_num}.json"
print("save to ", json_file)
rep_list = []  # 这个的作用是读取之前测试的结果，然后继续测试,如果之前的结果有的话，那么就直接跳过[:index]的测试数据
if os.path.exists(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        rep_list = json.load(file)
index = len(rep_list) # 如果index=0 测试所有数据
print("index:",index)
# 遍历数据
for item in tqdm(mme_data[index:], desc="Processing items"):
    # item 包含视频路径，视频类别，问题，以及答案
    video_path = os.path.join(data_path, item['url'] + ".mp4")
    print("===========================================================rep_list")
    print("video_path:",video_path)
    content = item.copy()
    frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
    raw_video = [f for f in frames]

    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    # 遍历不同的问题 (一个视频可能有多个问题)
    for q_num, question in enumerate(content['questions']):
        qs = ""
        qs += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + question['question'] + '\n' + " ".join(question['options']) + '\nThe best answer is:'
        print("======================================================")
        print("qs:")
        print(qs)
        res = llava_inference(qs, video)
        question['response'] = res # 原地修改 response
    print("len(rep_list)",len(rep_list))
    rep_list.append(content) # 在原数据上增加了 respondse
    print("len(rep_list)",len(rep_list))
    with open(json_file, "w", encoding='utf-8') as file:
        json.dump(rep_list, file, ensure_ascii=False, indent=4)
print(f"save result : {json_file}")
print("len(rep_list)",len(rep_list))
