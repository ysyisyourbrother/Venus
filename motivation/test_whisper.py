from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import torch
import os
import argparse
import sys
import re
sys.path.append('..')
from utils.audio import get_asr_docs,load_audio_model
from utils.rag import load_embedding_model,TextDataBase
import ffmpeg 
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser() 
    # Define the command-line arguments
    parser.add_argument("--model_path", default="/root/nfs/codespace/llm-models/MLLM/openai/whisper-large" )
    parser.add_argument("--bge_model_path", default="/root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base" )
    parser.add_argument("--video_path",  default="/root/nfs/codespace/Venus/demo/lake.mp4")
    return parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
whisper_model, whisper_processor = load_audio_model(args.model_path, device = device)
video_path =  args.video_path   
asr_docs_total = get_asr_docs(video_path, whisper_model, whisper_processor,chunk_length_s=10)
print(asr_docs_total)
print(len(asr_docs_total))
embedding_model = load_embedding_model(args.bge_model_path)
t_database = TextDataBase(embedding_model, 512)
t_database.add_documents(asr_docs_total)
t_database.print_index_ntotal()

query = "Describe the lake."
top_documents,idx = t_database.retrieve_documents_with_dynamic(query, threshold=0.4)
print("top_documents",top_documents)
print("idx",idx)