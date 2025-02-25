# run on server, test on dummy stream vedio
export CUDA_VISIBLE_DEVICES=0
python stream.py \
    --model-path /root/nfs/codespace/llm-models/MLLM/lmms-lab/LLaVA-Video-7B-Qwen2 \
    --output_name result \
    --video_path "./demo/xU25MMA2N4aVtYay.mp4" \
    --output_dir temp \
    --prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes." \
    --mm_spatial_pool_stride 4 \
    --for_get_frames_num 8 \
    --mm_spatial_pool_mode average \
    --mm_newline_position no_token \
    --overwrite true \
    --load_8bit true \
    --load_4bit false
  