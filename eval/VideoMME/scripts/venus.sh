export CUDA_VISIBLE_DEVICES=0,2 
python eval/VideoMME/eval_venus_videomme.py  \
--video_duration_type short \
--max_frames_num 32  \
--threshold 0.2  \
--top_k 20 \