import argparse
import torch
import time 
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
# import openai

from PIL import Image



import numpy as np

 
from decord import VideoReader, cpu
import numpy as np
class Dummy_Vedio:
    def __init__(self,video_path):
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)  # 获取视频的总帧数
        frame_idx = list(range(total_frame_num))  # 生成所有帧的索引
        all_frames = vr.get_batch(frame_idx).asnumpy()  # 获取帧并转换为NumPy数组
        self.all_frames = np.repeat(all_frames, 1, axis=0)  # 扩展第一个维度
        self.total_frame= self.all_frames.shape[0]
        self.index = 0
        print("total frame",self.total_frame)
        print("frame shape",self.all_frames[self.index].shape)
    def get_new_frame(self):
        if self.index < self.total_frame:
            frame = self.all_frames[self.index]
            self.index += 1
            return frame
        else:
            return None
        


 
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--load_4bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    return parser.parse_args()

 



 


def run_inference(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

        if "qwen" not in args.model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, 
                                                                               args.model_base, 
                                                                               model_name, 
                                                                                torch_dtype="bfloat16", 
                                                                                load_8bit=args.load_8bit, 
                                                                                load_4bit=args.load_4bit, 
                                                                                overwrite_config=overwrite_config,
                                                                            )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, 
                                                                               args.model_base, 
                                                                               model_name, 
                                                                               torch_dtype="bfloat16",
                                                                                load_8bit=args.load_8bit, 
                                                                                load_4bit=args.load_4bit, 
                                                                               )
    print(model.dtype)
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False


    question = args.prompt
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    print(qs)
    conv = conv_templates[ "qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    dummy_vedio = Dummy_Vedio( args.video_path)
    frame_queue = []
    gap = 1 # 一次query的帧数
    query_time = 0 # 循环推理次数
    while True:
        frame = dummy_vedio.get_new_frame()
        if frame is None:
            break
        frame_queue.append(frame)
        if len(frame_queue) == gap:
            vedio_clip = np.stack(frame_queue, axis=0) # 
            print("vedio_clip",vedio_clip.shape)
            start_time = time.perf_counter()
            video = image_processor.preprocess(vedio_clip, return_tensors="pt")["pixel_values"].half().cuda()
            end_time = time.perf_counter()
            print("image_processor time:", (end_time - start_time) * 1000, "ms")
            video = [video]
            print("video",video[0].shape)
            with torch.inference_mode():
                query_time += 1
                print("================================")
                start_time  = time.perf_counter()
                output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, 
                                        temperature=0.0, max_new_tokens=1024, top_p=0.1,num_beams=1,use_cache=True, stopping_criteria=[stopping_criteria])
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                end_time = time.perf_counter()
                print("query time:", (end_time - start_time) * 1000, "ms")
                print("decodeing token:",output_ids.shape )
                print("================================")
                print(f"Question: {prompt}\n")
                print("================================")
                print(f"Response: {outputs}\n")
                print("================================")
                print(torch.torch.cuda.max_memory_allocated()/1024/1024/1024,"GB")
            frame_queue = []
        if query_time == 3:
            break
    if len(frame_queue) > 0:
        vedio_clip = np.stack(frame_queue, axis=0)        


 
        
    


 
 

 


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

