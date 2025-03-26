export CUDA_VISIBLE_DEVICES=0,1
python eval/VideoMME/eval_venus_videomme.py  \
--video_duration_type short \
--max_frames_num 32  \
--threshold 0.2  \
--top_k 20 \
--vlm_path /root/nfs/codespace/llm-models/MLLM/lmms-lab/LLaVA-Video-7B-Qwen2 \
--bge_path /root/nfs/codespace/llm-models/MLLM/BAAI/BGE-VL-base \
--whisper_path /root/nfs/codespace/llm-models/MLLM/openai/whisper-large \
--yolo_path motivation/yolo11l.pt \
--mme_data_path /root/nfs/download/dataset/lmms-lab/Video-MME \ 
--filtered_frame_path /root/nfs/download/dataset/lmms-lab/short_frame \