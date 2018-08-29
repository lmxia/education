from django.conf import settings
from django.conf.urls import url
from views.line import LineView
__author__ = 'xialingming'

urlpatterns = [
    url(r'^line/?$', LineView.as_view(), name='line/'),
]