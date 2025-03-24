from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token,KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
import argparse
import copy
import numpy as np
import json
from tqdm import tqdm
import os
import easyocr
import time
from utils.vedio import process_video
from utils.rag import load_embedding_model,VideoDataBase,TextDataBase
from utils.audio import get_asr_docs,load_audio_model
from ultralytics import YOLO
import easyocr
def llava_inference(model, tokenizer,qs, video):
    if video is not None:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        question = qs
    else:
        question = qs
    print("======================================================")
    print("question:",question)
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, 
                    return_tensors="pt").unsqueeze(0).to(device)
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids, images=video, 
            attention_mask=attention_masks, 
            modalities="video", 
            do_sample=False, 
            temperature=0.0,
            max_new_tokens=16, #NOTE 对齐 generate_videomme.py
            top_p=0.1,
            num_beams=1, 
            use_cache=True)
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text_outputs
def save_json (data, path):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path, "w", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def save_frames(frames, path):
    file_paths = []
    if not os.path.exists(path):
        os.makedirs(path)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'{path}/frame_{i}.png'
        print("save path:",file_path)
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths
def get_filtered_frames(filter_frame_path, video_name):
    frame_dir = os.path.join(filter_frame_path, video_name)
    print("filter_frame_path",frame_dir)
    print("read filtered frames")
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"视频文件夹不存在: {frame_dir}")
    frame_files = ([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(('.jpg', '.png'))   
    ])
    frames = [ np.asarray(Image.open(image) ) for image in frame_files]
    print(len(frames))
    return frames
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Video with VideoMME settings.")
    parser.add_argument(
        "--video_duration_type",
        type=str,
        choices=["short", "medium", "long"],
        default="short",
        help="Specify the video duration type: short, medium, or long."
    )
    parser.add_argument(
        "--max_frames_num",
        required=True,
        type=int,
        help="Specify the maximum number of video frames to process."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Specify the threshold value.",
        default=0.2
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Specify the top_k value.",
        default=20
    )
    parser.add_argument(
        "--filtered",
        action='store_true',
        help= "Whether use filtered frames.",
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args() 
    max_frames_num =  args.max_frames_num
    threshold =  args.threshold
    asr_threshold = args.threshold
    top_k =  args.top_k
    device = "cuda"
    # save result 
    rag_data = dict()
    # load MME data
    data_path = "/root/nfs/download/dataset/lmms-lab/Video-MME" # mp4 data path
    filter_frame_path = "/root/nfs/download/dataset/lmms-lab/short_frame"
    with open("eval/short.json", 'r', encoding='utf-8') as file:
        mme_data = json.load(file)
    # choose on video from Video-MME, you cal also change to other video or your video file
    index =  35
    item = mme_data[index]
    video_path = os.path.join(data_path, item['url'] + ".mp4")
    # uniform sample frames for video
    frames, frame_time, video_time = process_video(video_path, max_frames_num,12, force_sample=True) #  
    save_frames([f for f in frames], "temp/video_frames_uniform_sampled")
    # frame filter
    if args.filtered: 
        frames = get_filtered_frames(filter_frame_path,item['url'])
    raw_video = [f for f in frames]
    
    # get question
    question = item['questions'][0]
    query = [question['question']]
    for o in question['options']:
        query.append(o)
    rag_data["content"] = item 
    rag_data["query"] = query
    #####################################################################
    # create video database
    embedding_model = load_embedding_model("/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base")
    yolo_model =  YOLO("motivation/yolo11l.pt").to(device)
    ocr_model =  easyocr.Reader(['en'])
    v_database = VideoDataBase(embedding_model, 512, yolo_model, ocr_model)
    v_database.add_frames(raw_video)
    v_database.print_index_ntotal()
    rag_data["ocr_list"] = v_database.ocr_list
    rag_data["det_list"] = v_database.det_list
    #####################################################################
    # create asr database
    whisper_model, whisper_processor = load_audio_model( "/root/nfs/codespace/llm-models/MLLM/openai/whisper-large", device = device)
    asr_docs_total = get_asr_docs(video_path, whisper_model, whisper_processor,chunk_length_s=5)
    print("len(asr_docs_total)",len(asr_docs_total))
    print("asr_docs_total",asr_docs_total)
    a_database = TextDataBase(embedding_model, 512)
    a_database.add_documents(asr_docs_total)
    a_database.print_index_ntotal()
    rag_data["asr_list"] = a_database.documents
    #####################################################################
    # retrieve video
    if not args.filtered:
        v_top_documents, v_idx = v_database.retrieve_documents_with_dynamic(query, threshold=threshold )
        if len(v_idx) == 0: #TODO: 存在选择帧数为0的情况
            v_top_documents, v_idx = v_database.retrieve_documents_top_k(query, top_k=top_k)
    else:
        v_top_documents, v_idx = v_database.retrieve_documents_top_k(query,top_k=20 )
    rag_data["v_idx"] = v_idx
    # retrieve asr
    a_top_documents,a_idx = a_database.retrieve_documents_with_dynamic(query, threshold=0.2)
    print("top_documents",a_top_documents)
    print("idx",a_idx)
    rag_data["a_idx"] = a_idx
    selected_frame = [ d["frame"] for d in v_top_documents] 
    if not args.filtered:
        save_frames( selected_frame, "temp/video_frames_uniform_selected")
    else:
        save_frames( selected_frame, "temp/video_frames_filter_selected")
    selected_det =  [ d["det"] for d in v_top_documents] 
    rag_data["selected_det"] = selected_det
    selected_det = [i for i in selected_det if i != ""]
    
    selected_ocr = [ d["ocr"] for d in v_top_documents]
    rag_data["ocr"] = selected_ocr 
    selected_ocr = [i for i in selected_ocr if i != ""]
    
    selected_asr = a_top_documents
    rag_data["selected_asr"] = selected_asr
    selected_asr = [i for i in selected_asr if i != ""]
    #TODO: 如何拼接
    rag_question =f"The video lasts for {video_time:.2f} seconds, and {str( len(selected_frame))} frames are selected from it."
    if len(selected_det) > 0:
        rag_question += f"\nVideo Object Detection information (given in chronological order of the video):  {'; '.join( selected_det)}"
    if len(selected_ocr) > 0:
        rag_question += "\nVideo OCR information (given in chronological order of the video): " + "; ".join( selected_ocr)
    if len(selected_asr) > 0:
        rag_question += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join( selected_asr)
    rag_question +="\nPlease answer the following questions related to this video."
    rag_question +=  "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + \
                        question['question'] + '\n' + " ".join(question['options']) + '\nThe best answer is:'
    print("rag_question",rag_question)

    #####################################################################
    # load your VLM
    print("load LLaVA-Video-7B-Qwen2.................................")
    device_map = {'model.vision_tower': 0, 'model.mm_projector': 0, 'model.norm': 0, 'model.rotary_emb': 0, 'model.embed_tokens': 0, 'model.image_newline': 0, 
                    'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 
                    'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 
                    'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 
                    'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2,  'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 
                    'lm_head': 2}    
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] =  "average"
    mem_before = torch.cuda.max_memory_allocated( )
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        "/root/nfs/codespace/llm-models/MLLM/lmms-lab/LLaVA-Video-7B-Qwen2", 
        None, 
        "llava_qwen", 
        torch_dtype="bfloat16", 
        load_in_8bit=False,
        load_in_4bit=False, 
        # device_map="auto",
        device_map=device_map,
        overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
    
    mem_after = torch.cuda.max_memory_allocated( )
    print("LLaVA-Video-7B-Qwen2 memory usage: {:.2f} GB".format((mem_after - mem_before) / 1024 / 1024/ 1024))
    model.eval()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    #################################################################
    # naive inference
    print("======================================================")
    print("inference with naive.................................")
    naive_question =  "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + \
                        question['question'] + '\n' + " ".join(question['options']) + '\nThe best answer is:'
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    print(len(video[0])) #  max frames
    print( (video[0].shape)) # torch.Size([max_frames, 3, 384, 384])
    text_outputs = llava_inference(model,tokenizer, naive_question, video)
    print(text_outputs)
    rag_data["naive_question"] = naive_question
    rag_data["naive_inference_answer"] = text_outputs
    ##################################################################
    # inference with RAG
    print("======================================================")
    print("inference with RAG.................................")
    selected_video = image_processor.preprocess(selected_frame, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    selected_video = [selected_video]
    text_outputs = llava_inference(model,tokenizer, rag_question, selected_video)
    rag_data["rag_question"] = rag_question
    rag_data["rag_inference_answer"] = text_outputs    
    save_json(rag_data,"temp/rag_data.json")
