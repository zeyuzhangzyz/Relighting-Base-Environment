import cv2
import numpy as np
import os
from PIL import Image, ImageFilter
import torch
from torch.autograd import Variable


def render_image(my_network,Lab, inputL, sh, save_path,row, col):
    outputImg, outputSH  = my_network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)

    resultLab = cv2.resize(resultLab, (row, col))
    cv2.imwrite(save_path, resultLab)


from camera import estimate_SH_coefficients, save_SH_coefficients, load_model, get_normal, get_shading

def mymain():
    for i in range(1,23):
        face_name = 'face2'
        current_directory = os.getcwd()
        print("当前工作目录：", current_directory)
        env_image = cv2.imread('img_useful/pic/env/%s.jpg' %i)
        env_image = cv2.resize(env_image, (512, 512))
        env_image = env_image / 255.0
        env_image = env_image.reshape(-1, 3)  # 如果 env_image 是三维数组
        normal = get_normal()
        sh_coefficients = estimate_SH_coefficients(env_image, normal)
        average_sh_coefficients = np.sum(sh_coefficients, axis=1)
        get_shading(normal, average_sh_coefficients, 'img_useful/pic/env/light'+str(i)+'.jpg')
        my_network = load_model('trained_model/trained_model_03.t7')
        img = cv2.imread('img_useful/pic/face/'+face_name+'.jpg')  # zaozao.jpg
        row, col, _ = img.shape
        img = cv2.resize(img, (512, 512))
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        inputL = Lab[:,:,0]
        inputL = inputL.astype(np.float32)/255.0
        inputL = inputL.transpose((0,1))
        inputL = inputL[None,None,...]
        inputL = Variable(torch.from_numpy(inputL).cuda())

        sh = average_sh_coefficients
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())

        des_img_name = 'img_useful/pic/env/'+face_name+'_relighted%s.jpg' %i
        render_image(my_network, Lab, inputL, sh, des_img_name,col, row)


mymain()
