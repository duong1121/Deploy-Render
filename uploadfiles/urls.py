from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name='index'),
    path("register/", views.UserRegisterView.as_view(), name='signup'),
    path("upload/", views.UploadView.as_view(), name='upload'),
    path("download/<int:id>/", views.DownloadView.as_view(), name='download'),
]
