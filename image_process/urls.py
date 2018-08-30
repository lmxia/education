from django.conf import settings
from django.conf.urls import url
from views.image import ImageView
__author__ = 'xialingming'

urlpatterns = [
    url(r'^process/?$', ImageView.as_view(), name='process/'),
]