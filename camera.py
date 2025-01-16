#以下为例子，因为终端无法desploy，所以保存视频帧
import cv2

rtsp_url = 'rtsp://admin:smc123456@192.168.123.67:554/stream1'
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error opening video stream or file")
else:
    print("Video stream opened successfully")
for i in range(10):
    ret, frame = cap.read()
    print(type(frame))
    print(frame.shape)
    print(type(ret))
    print(ret)
# 释放 VideoCapture 对象
cap.release()