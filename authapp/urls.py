from django.urls import path
from .views import (
    login_view,
    home_view,
    face_recognition_view,
    person_info_view,
    video_feed,
)


urlpatterns = [
    path('',login_view,name='login'),
    path('home/',home_view,name='home'),
    path('face_recognition/', face_recognition_view, name='face_recognition'),
    path('person_info/<int:person_id>/', person_info_view, name='person_info'),
    path('video_feed', video_feed, name='video_feed'),

]