from ultralytics import YOLO
import time
import os
import sys
sys.path.append('..')
from utils.vedio import process_video

# video_path = "../demo/animal.mp4"  # your video path
max_frames_num =   32
# frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True) #  
# raw_video = [f for f in frames]
# print( "len(raw_video)",len(raw_video))
images = ["cat.jpg"]

raw_video =  images*max_frames_num
model = YOLO("yolo11l.pt")
model.info()

start_time = time.perf_counter()
iter = 20
for i in range(iter):
    results = model( raw_video,stream=True,batch =max_frames_num)  # predict on an image
end_time = time.perf_counter()
print("=============================")
print(f"avg time for {max_frames_num} frames: {(end_time - start_time)/iter * 1000:.2f} ms")


total_name_list = []
total_con_list  = []
# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    confs = [c.item() for c in confs]
    total_name_list += names
    total_con_list += confs
    descriptions = [
            f"{name} located at (x1={coords[0]:.3f}, y1={coords[1]:.3f}, x2={coords[2]:.3f}, y2={coords[3]:.3f})"
            for name, conf, coords in zip(names, confs, xyxyn)
        ]    
    print(descriptions)
print(total_name_list),
print(total_con_list)