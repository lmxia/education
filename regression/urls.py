from django.conf import settings
from django.conf.urls import url
from views.line import LineView
from views.arc import ArcView
__author__ = 'xialingming'

urlpatterns = [
    url(r'^line/?$', LineView.as_view(), name='line/'),
    url(r'^arc/?$', ArcView.as_view(), name='arc/'),
]