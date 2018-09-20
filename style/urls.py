from django.conf import settings
from django.conf.urls import url
from . import views
__author__ = 'xialingming'

urlpatterns = [
    url(r'^transfer/?$', TransferView.as_view(), name='transfer/'), 
]