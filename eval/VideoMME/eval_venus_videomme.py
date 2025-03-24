from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token,KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
import copy
import numpy as np
import json
from tqdm import tqdm
import os
import easyocr
import time
from utils.vedio import process_video,get_filtered_frames
from utils.rag import load_embedding_model,VideoDataBase,TextDataBase
from utils.audio import get_asr_docs,load_audio_model
from ultralytics import YOLO
import easyocr
import argparse
def llava_inference(model, tokenizer,qs, video):
    if video is not None:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "Please answer the following questions related to this video.\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "Please answer the following questions related to this video.\n" + qs
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
        inputs=input_ids, 
        images=video, 
        attention_mask=attention_masks, 
        modalities="video", 
        do_sample=False, 
        temperature=0.0,
        max_new_tokens=16, #NOTE align with generate_videomme.py
        top_p=0.1,
        num_beams=1, 
        use_cache=True)
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text_outputs

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
    device = "cuda"
    args = parse_args() 
    max_frames_num =  args.max_frames_num
    threshold =  args.threshold
    asr_threshold = args.threshold
    top_k =  args.top_k
    # load your VLM
    print("load LLaVA-Video-7B-Qwen2.................................")
    # device_map = {'model.vision_tower': 0, 'model.mm_projector': 0, 'model.norm': 0, 'model.rotary_emb': 0, 'model.embed_tokens': 0, 'model.image_newline': 0, 
    #             'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 
    #             'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 
    #             'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 
    #             'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2,  'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 
    #             'lm_head': 2}
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
        # device_map= device_map,
        device_map="auto",
        overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
    # print(model)
    mem_after = torch.cuda.max_memory_allocated( )
    print("LLaVA-Video-7B-Qwen2 memory usage: {:.2f} GB".format((mem_after - mem_before) / 1024 / 1024/ 1024))
    model.eval()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # load other models
    device = "cuda"
    embedding_model = load_embedding_model("/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base", device = device)
    yolo_model =  YOLO("motivation/yolo11l.pt").to(device)
    ocr_model =  easyocr.Reader(['en']) #TODO: 是否要多语言
    whisper_model, whisper_processor = load_audio_model( "/root/nfs/codespace/llm-models/MLLM/openai/whisper-large", device = device)
    # load MME data
    data_path = "/root/nfs/download/dataset/lmms-lab/Video-MME" # mp4 data path
    filter_frame_path = "/root/nfs/download/dataset/lmms-lab/short_frame" # filtered frame path

    video_duration_type  =  args.video_duration_type
    with open(f"eval/VideoMME/{video_duration_type}.json", 'r', encoding='utf-8') as file:
        mme_data = json.load(file)
    print("total video number:", len(mme_data))
    # result path: save result to eval/results
    os.makedirs("eval/results", exist_ok=True)
    if args.filtered:
        result_file = f"eval/results/eval_venus_videomme_{video_duration_type}_{max_frames_num}_{threshold}_{top_k}_filtered.json"
    else:
        result_file = f"eval/results/eval_venus_videomme_{video_duration_type}_{max_frames_num}_{threshold}_{top_k}.json"
    rep_list = []
    if os.path.exists(result_file): # 若有结果文件，读取，从上一次中断的地方开始
        with open(result_file, 'r', encoding='utf-8') as file:
            rep_list = json.load(file)
    print("len(rep_list)",len(rep_list))
    index = len(rep_list)
    print("index:",index)
    
    # 遍历数据,从上一次中断的地方开始
    for item in tqdm(mme_data[index:], desc="Processing items"): 
        print("video id:",item['video_id'])
        if item['video_id'] == "500"   or item['video_id'] == "523" or item['video_id'] == "001" or item['video_id'] == "200":
            continue
        video_path = os.path.join(data_path, item['url'] + ".mp4")
        print("video_path",video_path)
        content = item.copy()    
        # sample frame uniformly from a video
        frames, frame_time, video_time = process_video(video_path, max_frames_num,12, force_sample=True) #  
        if args.filtered:
            # get filtered frames
            frames = get_filtered_frames(filter_frame_path,item['url'])
            content['filtered'] = True
            content["filtered_frames"] = len(frames)
        raw_video = [f for f in frames]
        print( "len(raw_video)",len(raw_video))
        #####################################################################
        # create video database
        v_database = VideoDataBase(embedding_model, 512, yolo_model, ocr_model)
        v_database.add_frames(raw_video)
        v_database.print_index_ntotal()
        #####################################################################
        # create asr database
        asr_docs_total = get_asr_docs(video_path, whisper_model, whisper_processor,chunk_length_s=5)
        print("len(asr_docs_total)",len(asr_docs_total))
        print("asr_docs_total",asr_docs_total)
        # TODO:: bge 只能处理 < 77 tokens #
        a_database = TextDataBase(embedding_model, 512)
        if len(asr_docs_total) != 0:
            a_database.add_documents( asr_docs_total )
        a_database.print_index_ntotal()
        #####################################################################
        # 遍历问题
        #TODO: 这里的 query是多个query list videorag 有处理多个query list的实现 type query <class 'list'>
        # bge encoder [list str]
        for q_num, question in enumerate(content['questions']): 
            query = [question['question']]
            for o in question['options']:
                query.append(o)
            # retrieve video
            print("type query",type(query)) 
            print("query",query)
            #TODO: 问题超过77 tokens
            if not args.filtered:
                v_top_documents, v_idx = v_database.retrieve_documents_with_dynamic(query, threshold=threshold )
                if len(v_idx) == 0: #TODO: 存在选择帧数为0的情况
                    v_top_documents, v_idx = v_database.retrieve_documents_top_k(query, top_k=top_k)
            else:
                v_top_documents, v_idx = v_database.retrieve_documents_top_k(query, top_k=top_k)
            selected_frame = [ d["frame"] for d in v_top_documents] 
            selected_det =  [ d["det"] for d in v_top_documents] 
            selected_det = [i for i in selected_det if i != ""]
            selected_ocr = [ d["ocr"] for d in v_top_documents] 
            selected_ocr = [i for i in selected_ocr if i != ""]
            # retrieve asr
            a_top_documents,a_idx = a_database.retrieve_documents_with_dynamic(query, threshold= asr_threshold)      
            selected_asr = a_top_documents
            selected_asr = [i for i in selected_asr if i != ""]
            print("selected_video",len(selected_frame))
            print("selected_det",len(selected_det))
            print("selected_ocr",len(selected_ocr))
            print("selected_asr",len(selected_asr))
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
            question["rag_question"] = rag_question
            question["select_frame_num"] = len(selected_frame)
            # inference with RAG
            print("======================================================")
            print("inference with RAG.................................")
            selected_video = image_processor.preprocess(selected_frame, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            selected_video = [selected_video]
            res = llava_inference(model,tokenizer, rag_question, selected_video)
            print(res)
            question['response'] = res # 原地修改
        # content 增加 selected_video num
        rep_list.append(content) # 在原数据上增加了 response
        print("content",content)
        # 每次得到一个结果，就保存一次
        with open(result_file, "w", encoding='utf-8') as file:
            json.dump(rep_list, file, ensure_ascii=False, indent=4)
        print(f"save result to...: {result_file}")