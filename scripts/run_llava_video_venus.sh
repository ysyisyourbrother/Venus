export CUDA_VISIBLE_DEVICES=0,1 
python  venus_demo.py \
    --max_frames_num 32 \
    --threshold 0.2 \
    --top_k 20 \
    --video_duration_type short \
    --attn_implementation "eager" 


