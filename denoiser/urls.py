from django.urls import path
from django.contrib.auth import views as django_auth_views
from . import views
from . import auth_views
from . import profile_views

app_name = 'denoiser'

urlpatterns = [
    path('', views.index, name='home'),
    path('processor/', views.audio_processor, name='audio_processor'),
    path('test-audio/<str:folder_name>/', views.test_audio, name='test_audio'),
    path('test-audio-original/<str:folder_name>/', views.test_audio_original, name='test_audio_original'),
    path('debug-session/', views.debug_session, name='debug_session'),
    path('progress/<str:task_id>/', views.progress_page, name='progress_page'),
    path('api/progress/<str:task_id>/', views.get_progress, name='get_progress'),
    path('results/<str:folder_name>/', views.results_page, name='results_page'),
    
    # Authentication URLs
    path('login/', auth_views.login_view, name='login'),
    path('register/', auth_views.register_view, name='register'),
    path('logout/', auth_views.logout_confirm, name='logout'),
    path('api/check-auth/', auth_views.check_auth_status, name='check_auth'),
    
    # Profile URLs
    path('profile/', profile_views.profile_view, name='profile'),
    path('profile/edit/', profile_views.edit_profile_view, name='edit_profile'),
    path('profile/change-password/', profile_views.change_password_view, name='change_password'),
    path('profile/audio-history/', profile_views.audio_history_view, name='audio_history'),
    path('profile/delete-upload/<int:upload_id>/', profile_views.delete_audio_upload, name='delete_audio_upload'),
    path('profile/download/<int:upload_id>/<str:file_type>/', profile_views.download_audio_file, name='download_audio_file'),
]
