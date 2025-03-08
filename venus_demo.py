from PIL import Image
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
from utils.vedio import process_video
from utils.rag import load_embedding_model,VideoDataBase,TextDataBase
from utils.audio import get_asr_docs,load_audio_model
from ultralytics import YOLO
import easyocr
def llava_inference(model, tokenizer,qs, video):
    #TODO: 对齐 
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
    conv = conv_templates[ "qwen_1_5"].copy()
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
            inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", 
                                        do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True)
      
   
    text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return text_outputs

if __name__ == "__main__":
    device = "cuda"
    # load vedio
    max_frames_num =  16
    video_path = "/root/nfs/codespace/Venus/demo/lake.mp4"  # your video path
    question = "Describe the lake."  # your question
    # frame filter
    frames, frame_time, video_time = process_video(video_path, max_frames_num,12, force_sample=True) #  
    raw_video = [f for f in frames]
    print( "len(raw_video)",len(raw_video))
    #####################################################################
    # create video database
    embedding_model = load_embedding_model("/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base")
    yolo_model =  YOLO("motivation/yolo11l.pt").to(device)
    ocr_model =  easyocr.Reader(['en'])
    v_database = VideoDataBase(embedding_model, 512, yolo_model, ocr_model)
    v_database.add_frames(raw_video)
    v_database.print_index_ntotal()
    #####################################################################
    # create asr database
    whisper_model, whisper_processor = load_audio_model( "/root/nfs/codespace/llm-models/MLLM/openai/whisper-large", device = device)
    asr_docs_total = get_asr_docs(video_path, whisper_model, whisper_processor)
    print("len(asr_docs_total)",len(asr_docs_total))
    print("asr_docs_total",asr_docs_total)
    a_database = TextDataBase(embedding_model, 512)
    a_database.add_documents(asr_docs_total)
    a_database.print_index_ntotal()
    #####################################################################
    # retrieve video
    v_top_documents, v_idx = v_database.retrieve_documents_with_dynamic(question, threshold=0.4)
    print("idx",v_idx)
    # retrieve asr
    query =  question
    a_top_documents,a_idx = a_database.retrieve_documents_with_dynamic(query, threshold=0.4)
    print("top_documents",a_top_documents)
    print("idx",a_idx)

    selected_frame = [ d["frame"] for d in v_top_documents] 
    selected_det =  [ d["det"] for d in v_top_documents] 
    selected_det = [i for i in selected_det if i != ""]
    selected_ocr = [ d["ocr"] for d in v_top_documents] 
    selected_ocr = [i for i in selected_ocr if i != ""]
    selected_asr = a_top_documents
    selected_asr = [i for i in selected_asr if i != ""]
    print("selected_video",len(selected_frame))
    print("selected_det",len(selected_det))
    print("selected_ocr",len(selected_ocr))
    print("selected_asr",len(selected_asr))
    #TODO: 如何拼接
    rag_question =f"\nVideo have selected {str( len(selected_frame))} frames in total"
    rag_question+= f"\nVideo Object Detection information (given in chronological order of the video):  {'; '.join( selected_det)}"
    rag_question += "\nVideo OCR information (given in chronological order of the video): " + "; ".join( selected_ocr)
    rag_question += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join( selected_asr)
    rag_question += "\nPlease answer the following questions related to this video: " + question 
    print("rag_question",rag_question)
    #####################################################################
    # load your VLM
    print("load LLaVA-Video-7B-Qwen2.................................")
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] =  "average"
    mem_before = torch.cuda.max_memory_allocated( )
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        "/root/nfs/codespace/llm-models/MLLM/lmms-lab/LLaVA-Video-7B-Qwen2", 
        None, 
        "llava_qwen", 
        torch_dtype="bfloat16", 
        load_in_8bit=False,
        load_in_4bit=True, 
        overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
    mem_after = torch.cuda.max_memory_allocated( )
    print("LLaVA-Video-7B-Qwen2 memory usage: {:.2f} GB".format((mem_after - mem_before) / 1024 / 1024/ 1024))
    model.eval()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    #################################################################
    # naive inference
    print("======================================================")
    print("inference with naive.................................")
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    print(len(video[0])) #  max frames
    print( (video[0].shape)) # torch.Size([max_frames, 3, 384, 384])
    text_outputs = llava_inference(model,tokenizer, question, video)
    print(text_outputs)
    ##################################################################
    # inference with RAG
    print("======================================================")
    print("inference with RAG.................................")
    selected_video = image_processor.preprocess(selected_frame, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    selected_video = [selected_video]
    print(len(selected_video[0])) #  
    print( (selected_video[0].shape)) # 
    text_outputs = llava_inference(model,tokenizer, rag_question, selected_video)
    print(text_outputs)
