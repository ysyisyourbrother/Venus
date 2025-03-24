from decord import VideoReader, cpu
import numpy as np
import os
import ffmpeg
from PIL import Image
def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    # TODO: 这里可以加上抽帧算法
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time
def get_filtered_frames(filter_frame_path, video_name):
    frame_dir = os.path.join(filter_frame_path, video_name)
    print("filter_frame_path",frame_dir)
    print("read filtered frames")
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"视频文件夹不存在: {frame_dir}")
    frame_files = ([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(('.jpg', '.png'))   
    ])
    frames = [ np.asarray(Image.open(image) ) for image in frame_files]
    print(len(frames))
    return frames