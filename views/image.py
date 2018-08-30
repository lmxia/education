from rest_framework.views import APIView
from controls.image import ImageControl
from django.http import QueryDict
from rest_framework.request import Request
from rest_framework.parsers import FileUploadParser

class ImageView(APIView):
    def post(self, request, *args, **kwargs):
        parser_classes = (FileUploadParser,)
        file_obj = request.data['file']
        return ImageControl.load_image(file_obj)
