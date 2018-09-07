from rest_framework.views import APIView
from controls.line import LineControl
from django.http import QueryDict
from rest_framework.request import Request

class ArcView(APIView):
    def post(self, request, *args, **kwargs):
        params=get_parameter_dic(request)
        return ArcControl.regression(params.get("points"))
