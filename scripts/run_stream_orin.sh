# run on orin, test on strewam vedio from camera
python stream_orin.py  \
--model-path /home/eco/CodeSpace/model/LLaVA-Video-7B-Qwen2  \
--output_name result  \
--video_path "rtsp://admin:smc123456@192.168.123.67:554/stream1" \
--output_dir temp  \
--prompt "Please provide a detailed description of the video, focusing on the main subjects, their actions, the background scenes."  \
--mm_spatial_pool_stride 4  \
--for_get_frames_num 1  \
--mm_spatial_pool_mode average \
--mm_newline_position no_token \
--overwrite true \
--load_8bit false \
--load_4bit true
  