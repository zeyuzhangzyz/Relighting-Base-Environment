简体中文 | [English](README_EN.md)



我们基于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)的分割模型和[DPR](https://github.com/zhhoper/DPR)算法，实现了一个前后端交互的效果的relighting课程设计项目。



**基本目标：**

完成离线模块的核心算法部分，实现场景光照的提取，实现人像的重打光。



**拓展目标：**

完成UI客户端设计，能够通过UI与用户交互，上传图片。

服务端部分能够接收用户请求，保存图片。

服务端能够顺利调用核心图像拼接算法。

客户端、服务端、核心算法顺利联调，成功运行，无明显恶性bug



![image-20231031235159511](image-20231031235159511.png)



使用方法：安装[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)和[DPR](https://github.com/zhhoper/DPR)所需要的所有库函数或者requirements

conda create --Paddle --requirements

需要注意的是百度Paddle版本与CUDA和pytorch版本需要匹配。



使用方法，在根目录下  python manage.py runserver



针对人像视频的relighting：未部署到前端，可单独运行算法，运行video_relighting即可。









