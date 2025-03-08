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


 

if __name__ == "__main__":

    device = "cuda"
    max_frames_num =  32

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
        # device_map="auto",
        overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args
    mem_after = torch.cuda.max_memory_allocated( )
    print("LLaVA-Video-7B-Qwen2 memory usage: {:.2f} GB".format((mem_after - mem_before) / 1024 / 1024/ 1024))
    model.eval()
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # load other models
    embedding_model = load_embedding_model("/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base", device = device)
    yolo_model =  YOLO("motivation/yolo11l.pt").to(device)
    ocr_model =  easyocr.Reader(['en']) #TODO: 是否要多语言
    whisper_model, whisper_processor = load_audio_model( "/root/nfs/codespace/llm-models/MLLM/openai/whisper-large", device = device)
    # load MME data
    data_path = "/root/nfs/download/dataset/lmms-lab/Video-MME" # mp4 data path
    with open("eval/short.json", 'r', encoding='utf-8') as file:
        mme_data = json.load(file)
    print("total video number:", len(mme_data))
    # result path: save result to eval/results
    os.makedirs("eval/results", exist_ok=True)
    result_file = f"eval/results/eval_venus_videomme.json"
    rep_list = [] # 初始化
    # 遍历数据
    for item in tqdm(mme_data[0:], desc="Processing items"):
        video_path = os.path.join(data_path, item['url'] + ".mp4")
        content = item.copy()    
        # frame filter add here 
        frames, frame_time, video_time = process_video(video_path, max_frames_num,12, force_sample=True) #  
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
        asr_docs_total = [ doc[:75] for doc in asr_docs_total]
        a_database = TextDataBase(embedding_model, 512)
        a_database.add_documents( asr_docs_total[:5])
        a_database.print_index_ntotal()
        #####################################################################
        # 遍历问题
        for q_num, question in enumerate(content['questions']): 
            query = [question['question']]
            for o in question['options']:
                query.append(o)
            # retrieve video
            v_top_documents, v_idx = v_database.retrieve_documents_with_dynamic(query, threshold=0.1)
            selected_frame = [ d["frame"] for d in v_top_documents] 
            selected_det =  [ d["det"] for d in v_top_documents] 
            selected_det = [i for i in selected_det if i != ""]
            selected_ocr = [ d["ocr"] for d in v_top_documents] 
            selected_ocr = [i for i in selected_ocr if i != ""]
            # retrieve asr
            a_top_documents,a_idx = a_database.retrieve_documents_with_dynamic(query, threshold=0.1)      
            selected_asr = a_top_documents
            selected_asr = [i for i in selected_asr if i != ""]
            print("selected_video",len(selected_frame))
            print("selected_det",len(selected_det))
            print("selected_ocr",len(selected_ocr))
            print("selected_asr",len(selected_asr))
            #TODO: 如何拼接
            rag_question =f"\nVideo have selected {str( len(selected_frame))} frames in total."
            if len(selected_det) > 0:
                rag_question += f"\nVideo Object Detection information (given in chronological order of the video):  {'; '.join( selected_det)}"
            if len(selected_ocr) > 0:
                rag_question += "\nVideo OCR information (given in chronological order of the video): " + "; ".join( selected_ocr)
            if len(selected_asr) > 0:
                rag_question += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join( selected_asr)
            rag_question +=  "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + \
                                question['question'] + '\n' + " ".join(question['options']) + '\nThe best answer is:'
            print("rag_question",rag_question)
            # inference with RAG
            print("======================================================")
            print("inference with RAG.................................")
            selected_video = image_processor.preprocess(selected_frame, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            selected_video = [selected_video]
            print(len(selected_video[0])) #  
            print( (selected_video[0].shape)) # 
            res = llava_inference(model,tokenizer, rag_question, selected_video)
            print(res)
            question['response'] = res # 原地修改
        rep_list.append(content) # 在原数据上增加了 response
    print("len(rep_list)",rep_list)
    with open(result_file, "w", encoding='utf-8') as file:
        json.dump(rep_list, file, ensure_ascii=False, indent=4)
    print(f"save result to...: {result_file}")