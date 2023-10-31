import cv2
import os

# 输入图像文件夹和文件名模式
image_folder = 'img/'
video_name = 'img/output_video.mp4'

# 获取图像文件列表
images = ['result%s.jpg' %i for i in range(100)]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 使用VideoWriter编码视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
video = cv2.VideoWriter(video_name, fourcc, 20, (width, height))  # '30' 是帧率

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
