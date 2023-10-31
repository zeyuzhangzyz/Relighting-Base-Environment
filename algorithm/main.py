from base_kit import *



def mymain():
    """
    Perform relighting on a single image.
    """
    # Predict the segmentation result of the face
    face_name = 'face'
    env_name = 'bgi'
    command = 'conda activate Paddle & python tools/predict.py        ' \
          '--config configs/human_pp_humansegv2_lite.yml      ' \
          '--model_path pretrained_models/human_pp_humansegv2_lite_192x192_pretrained/model.pdparams        ' \
          '--image_path img/'+face_name+'.jpg' \
              ' --save_dir img'



    os.system(command)


    image_path = 'img/pseudo_color_prediction/'+face_name+'.png'
    mask = create_mask(image_path)

    a = -100
    b = 50

    a = 0
    b = 0
    truth_image_path = 'img/'+face_name+'.jpg'
    truth_image = cv2.imread(truth_image_path)
    width,height = truth_image.shape[0],truth_image.shape[1]
    truth_image, mask = shift_image(truth_image, mask, a, b)

    cv2.imwrite('img/'+face_name+'_shift.jpg',truth_image)
    cv2.imwrite('img/mask_shift.jpg',mask)
    # main.py



    env_image = cv2.imread('img/'+env_name+'.jpg')
    env_image = cv2.resize(env_image, (512, 512))
    env_image = env_image / 255.0
    env_image = env_image.reshape(-1, 3)
    normal = get_normal()
    sh_coefficients = estimate_SH_coefficients(env_image, normal)
    average_sh_coefficients = np.sum(sh_coefficients, axis=1)
    get_shading(normal, average_sh_coefficients, 'img/light.jpg')
    txt_name = 'light.txt'
    file_path = 'data/light/'+ txt_name
    save_SH_coefficients(file_path, average_sh_coefficients)

    my_network = load_model('trained_model/trained_model_03.t7')

    lightFolder = 'data/light/'

    img = cv2.imread('img/'+face_name+'_shift.jpg')  # zaozao.jpg
    row, col, _ = img.shape
    img = cv2.resize(img, (512, 512))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    sh = np.loadtxt(os.path.join(lightFolder, 'light.txt'))
    sh = sh[0:9]
    sh = sh
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())

    des_img_name = 'img/'+face_name+'_relighted.jpg'
    render_image(my_network, Lab, inputL, sh, des_img_name,height,width)

    face_image = cv2.imread(des_img_name)
    face_image = cv2.resize(face_image, (height,width))

    env_image_path = 'img/'+env_name+'.jpg'

    env_image = cv2.imread(env_image_path)
    env_image = cv2.resize(env_image, (height,width))

    merged_image = np.where(mask == 1, face_image, env_image)
    cv2.imwrite('img/result.jpg',merged_image)

    img = cv2.resize(img, (height,width))
    merged_image = np.where(mask == 1, img, env_image)
    cv2.imwrite('img/result_only_merge.jpg',merged_image)

