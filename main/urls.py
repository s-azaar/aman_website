# urls.py
from django.urls import path

from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('401/',views.page401,name='401'),
    path('logout/',views.logout,name='logout'),
    path('addnewcamera/',views.addnewcamera,name='addnewcamera'),
    path('video_result/', views.video_result, name='video_result'),
    path('camera_result/', views.camera_result, name='camera_result'),



]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)