import sys
import os
import cv2
from utils.utils_SH import SH_basis
from model.defineHourglass_512_gray_skip import *
from PIL import Image
sys.path.append('model')
sys.path.append('utils')
sys.path.append('trained_model')
modelFolder = 'trained_model/'

def estimate_SH_coefficients(env_image, normals):
    """
    Estimate the spherical harmonics (SH) coefficients given the environment image and normals.

    Parameters:
    - env_image: The environment image.
    - normals: The normals.

    Returns:
    - sh_coefficients: The SH coefficients.
    """
    sh_basis = SH_basis(normals)
    sh_coefficients = np.linalg.lstsq(sh_basis, env_image, rcond=None)[0]

    return sh_coefficients

def save_SH_coefficients(file_path, sh_coefficients):
    """
    Save the SH coefficients to a file.

    Parameters:
    - file_path: The file path to save the SH coefficients.
    - sh_coefficients: The SH coefficients.
    """
    np.savetxt(file_path, sh_coefficients, delimiter='\n', fmt='%.6f')

def load_model(model_path):
    """
    Load the trained model.

    Parameters:
    - model_path: The path to the trained model.

    Returns:
    - my_network: The loaded model.
    """
    my_network = HourglassNet()
    my_network.load_state_dict(torch.load(model_path))
    my_network.cuda()
    my_network.train(False)
    return my_network

def render_image(my_network, Lab, inputL, sh, save_path, height, width):
    """
    Render the image using the trained model.

    Parameters:
    - my_network: The trained model.
    - Lab: The LAB image.
    - inputL: The L channel of the LAB image.
    - sh: The SH coefficients.
    - save_path: The path to save the rendered image.
    - height: The height of the rendered image.
    - width: The width of the rendered image.
    """
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
    """
    Generate the normal map.

    Returns:
    - normal: The normal map.
    """
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
    """
    Estimate the spherical harmonics (SH) coefficients given the environment image and normals.

    Parameters:
    - env_image: The environment image.
    - normals: The normals.

    Returns:
    - sh_coefficients: The SH coefficients.
    """
    # Compute the spherical harmonics basis functions for the entire image
    sh_basis = SH_basis(normals)

    # Use least squares to fit the SH coefficients
    sh_coefficients = np.linalg.lstsq(sh_basis, env_image, rcond=None)[0]

    return sh_coefficients

def get_shading(normal, SH, save_path):
    '''
    Get shading based on normals and SH.

    Parameters:
    - normal: Nx3 matrix representing the normals.
    - SH: 9 x m vector representing the SH coefficients.

    Returns:
    - shading: Nxm vector representing the shading.
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)

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

def extract_light():
    for i in range(1,23):
        face_name = 'face2'
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)
        env_image = cv2.imread('img_useful/pic/env/%s.jpg' %i)
        env_image = cv2.resize(env_image, (512, 512))
        env_image = env_image / 255.0
        env_image = env_image.reshape(-1, 3)  # If env_image is a 3D array
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

def resize_image(input_image_path, output_image_path, size=(512, 512)):
    """
    Resize the input image to the specified size and save it to the output path.

    Parameters:
    - input_image_path: The path of the input image.
    - output_image_path: The path to save the output image.
    - size: The size of the resized image. Default is (512, 512).
    """
    image = Image.open(input_image_path)
    resized_image = image.resize(size)
    resized_image.save(output_image_path)

def create_mask(image_path):
    """
    Create a mask based on the image path.

    Parameters:
    - image_path: The path of the image.

    Returns:
    - mask2: The mask image.
    """
    image = cv2.imread(image_path)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[image[:,:,1] == 128] = 1
    mask2 = np.zeros((image.shape[0], image.shape[1],3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 1:
                mask2[i][j][0] = 1
                mask2[i][j][1] = 1
                mask2[i][j][2] = 1
    return mask2

def shift_image(image, mask, a, b):
    """
    Shift the image and mask vertically and horizontally.

    Parameters:
    - image: The input image.
    - mask: The input mask.
    - a: The vertical shift amount.
    - b: The horizontal shift amount.

    Returns:
    - image: The shifted image.
    - mask: The shifted mask.
    """
    image = np.roll(image, a, axis=0)
    mask = np.roll(mask, a, axis=0)
    image = np.roll(image, b, axis=1)
    mask = np.roll(mask, b, axis=1)
    return image, mask
