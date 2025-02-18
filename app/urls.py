"""heart URL Configuration

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
from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home',views.home),
    path('',views.index),
    path('index',views.index),
    path('register/',views.register),
    path('register/addregister',views.addregister),
    path('login/',views.login),
    path('login/addlogin',views.addlogin),
    path('logout/',views.logout),
    path('viewuser/',views.viewuser),
    path('upload/',views.upload),
    path('upload/addupload',views.addupload),
    path('viewupload/',views.viewupload),
    path('prediction/',views.prediction),
    path('prediction/predict',views.predict),
    path('attend_interview',views.attend_interview),
    path('start_interview',views.start_interview,name = "start_interview"),
    path('stop_interview', views.stop_interview, name='stop_interview'),
    path('interview_analysis', views.interview_analysis, name='interview_analysis'),
    path('view_results/<int:id>', views.view_results, name='view_results'),
    path('speech_recognition',views.speech_recognition),
    path('result',views.viewresult),
   

    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
