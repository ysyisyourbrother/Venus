# run on orin, test on demo vedio
python video_demo.py  \
--model-path /home/eco/CodeSpace/model/LLaVA-Video-7B-Qwen2  \
--output_name result  \
--video_path "./demo/xU25MMA2N4aVtYay.mp4" \
--output_dir temp  \
--prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."  \
--mm_spatial_pool_stride 4  \
--for_get_frames_num 2  \
--mm_spatial_pool_mode average \
--mm_newline_position no_token \
--overwrite true \
--load_8bit false \
--load_4bit false