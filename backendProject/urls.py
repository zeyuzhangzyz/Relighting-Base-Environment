"""backendProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from app.views import face
from app.views import bgi
from app.views import get_image
from app.views import root_redirect
from django.shortcuts import redirect
urlpatterns = [
    path('api/face', face, name='face'),
    path('api/bgi', bgi, name='bgi'),
    path('api/image', get_image, name='get_image'),
    path('', root_redirect,name='root')
]
