import cv2

# 打开视频文件
video_path = "img/input.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 逐帧提取并保存
frame_count = 0
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        break
    if frame_count>=1300 and frame_count<1400:
    # 生成帧文件名
        frame_filename = f"img/env{frame_count:04d}.jpg"

        # 保存帧为jpg文件
        cv2.imwrite(frame_filename, frame)

    # 增加帧计数器
    frame_count += 1

# 释放视频文件和OpenCV窗口
cap.release()
cv2.destroyAllWindows()