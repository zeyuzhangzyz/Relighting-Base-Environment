Translated to English: [简体中文](README_CN.md) | English

We have implemented a relighting course design project with interactive front-end and back-end effects based on the segmentation model of [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) and the algorithm of [DPR](https://github.com/zhhoper/DPR).

**Primary Goals:**

Complete the core algorithm of the offline module to extract scene lighting and perform relighting on portraits.

**Extended Goals:**

Design a UI client that allows users to interact and upload images through the UI.

The server should be able to receive user requests and save images.

The server should be able to successfully invoke the core image stitching algorithm.

The client, server, and core algorithm should be seamlessly integrated and run successfully without major bugs.

Demonstration:

![image-20231031235159511](image-20231031235159511.png)

**Environment Setup:**

Install all the required libraries for [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) and [DPR](https://github.com/zhhoper/DPR).

First, create a virtual environment using conda, which already includes many pre-installed packages. Install the libraries listed in the requirement.txt file, and then install PyTorch. Depending on your CUDA version (e.g., CUDA=11.7) and the corresponding cuDNN version, use the following commands to install the appropriate torch ... ... ... and paddlepaddle versions (please replace with the corresponding versions; Paddle version should match CUDA and PyTorch versions. Refer to [PyTorch](https://pytorch.org/) and [Windows PIP Installation - Documentation - PaddlePaddle Deep Learning Platform](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html)). For CPU versions, install accordingly. (If the installation of torch corresponding to CUDA=11.7 fails frequently, you can also use 11.8)

conda ... ... ... create -n Paddle python=3.9.12

pip install -r requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

python -m pip install ... ... ... paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

**Usage**: In the root directory, run python manage.py ... ... ... runserver.

Alternatively, run the video_relighting.py function and select the video and image manually.

**Warning:** If you encounter the error "Library cublas64_12.dll is not found," please copy the file Library cublas64_11.dll in the environment's bin directory and rename it to Library cublas64_12.dll. The reason for this error is currently unknown.

Disclaimer: This project references the code from [DPR](https://github.com/zhhoper/DPR).
