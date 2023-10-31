import cv2
import numpy as np
import os
from PIL import Image
# 指定图片路径

def resize_image(image_path, output_path, size=(512, 512)):
    image = Image.open(image_path)
    resized_image = image.resize(size)
    resized_image.save(output_path)
# 示例用法
input_image_path = ('img/face.jpg')  # 输入图片路径
output_image_path = ('img/face_resize.jpg')  # 输出图片路径
resize_image(input_image_path, output_image_path)



image_path = 'output/result/pseudo_color_prediction/photo_resize.png'
# 读取图片
image = cv2.imread(image_path)

# mask = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)

mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

# 设置对应像素点的值为 128
# mask[image[:,:,1] == 128][0] = 1
# mask[image[:,:,1] == 128][1] = 1
# mask[image[:,:,1] == 128][2] = 1
mask[image[:,:,1] == 128] = 1

mask2 = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i][j] == 1:
            mask2[i][j][0] = 1
            mask2[i][j][1] = 1
            mask2[i][j][2] = 1
mask = mask2



a = -100
b = 50
# 读取图片
truth_image = cv2.imread('data/mydata/photo_resize.jpg')
height, width, _ = truth_image.shape
# 上下平移
truth_image1 = np.roll(truth_image, a, axis=0)
mask = np.roll(mask, a, axis=0)
# 左右平移
truth_image2 = np.roll(truth_image1, b, axis=1)
mask = np.roll(mask, b, axis=1)



cv2.imwrite('data/mydata/photo2.jpg',truth_image2)
command = 'python camera.py'
# print(command)
os.system(command)
index = 3

env_img_name = 'env3.png'

env_image = cv2.imread('data/mydata/' + env_img_name)
truth_image = cv2.imread('result/photo2_03.jpg')
merged_image = np.where(mask == 1, truth_image, env_image)

# 显示合并后的图片
cv2.imshow('Merged Image', merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# saveFolder = 'result.photo'
# cv2.imwrite(os.path.join(saveFolder, human_img[:-4] + '_{:02d}.jpg'.format(index)), resultLab)
# print(image)
# # 检查是否成功读取图片
# if image is not None:
#     # 在这里进行您想要的操作，比如展示图片或进行图像处理
#     cv2.imshow('Image', mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print('无法读取图片')



