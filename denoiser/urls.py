from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('processor/', views.audio_processor, name='audio_processor'),
]
