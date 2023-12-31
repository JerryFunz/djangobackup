"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from app01 import views
from app02.views import read_data_from_wechat

urlpatterns = [
    # path("index/", views.index),
    # path("user_list/", views.user_list),
    # 接受小程序的传参
    path("pred/", views.pred),
    path("calc_accuracy/", views.calc_accuracy),
    path('import-data/', read_data_from_wechat, name='import_data'),
    # 其他URL配置...

]
