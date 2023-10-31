'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')
sys.path.append('trained_model')

modelFolder = 'trained_model/'


import cv2
from utils.utils_SH import SH_basis
from model.defineHourglass_512_gray_skip import *

def estimate_SH_coefficients(env_image, normals):
    sh_basis = SH_basis(normals)
    sh_coefficients = np.linalg.lstsq(sh_basis, env_image, rcond=None)[0]

    return sh_coefficients

def save_SH_coefficients(file_path, sh_coefficients):
    np.savetxt(file_path, sh_coefficients, delimiter='\n', fmt='%.6f')

def load_model(model_path):
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(model_path))
    my_network.cuda()
    my_network.train(False)
    return my_network

def render_image(my_network,Lab, inputL, sh, save_path,height,width):
    outputImg, outputSH  = my_network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)

    resultLab = cv2.resize(resultLab, (height,width))
    cv2.imwrite(save_path, resultLab)

def get_normal():
    img_size = 512
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)
    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    return normal
#-----------------------------------------------------------------

def estimate_SH_coefficients(env_image, normals):
    # 计算整个图像的球谐基函数
    sh_basis = SH_basis(normals)

    # 使用最小二乘法拟合球谐系数
    sh_coefficients = np.linalg.lstsq(sh_basis, env_image, rcond=None)[0]

    return sh_coefficients



def get_shading(normal, SH,save_path):
    '''
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    #shading = np.matmul(np.reshape(sh_basis, (-1, 9)), SH)
    #shading = np.reshape(shading, normal.shape[0:2])

    img_size = 512
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1


    sh = SH
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (512, 512))
    shading = shading * valid
    cv2.imwrite(save_path, shading)