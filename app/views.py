from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def face(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        save_path = 'img/face.jpg'  # 替换为你想要保存图像的目录及文件名
        with open(save_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        return JsonResponse({'message': 'Image uploaded successfully.'})
    else:
        return JsonResponse({'message': 'Invalid request.'}, status=400)

@csrf_exempt
def bgi(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        save_path = 'img/bgi.jpg'  # 替换为你想要保存图像的目录及文件名
        with open(save_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        return JsonResponse({'message': 'Image uploaded successfully.'})
    else:
        return JsonResponse({'message': 'Invalid request.'}, status=400)

from django.templatetags.static import static
from django.http import HttpResponse
from algorithm.main import mymain
def get_image(request):
    mymain()
    image_path = 'img/result.jpg'  # 替换为你要传递给前端的图片路径
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return HttpResponse(image_data, content_type='image/jpeg')


def root_redirect(request):
    return HttpResponse(status=302, content='', headers={'Location': '/html/index.html'})
