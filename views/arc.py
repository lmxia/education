from rest_framework.views import APIView
from controls.arc import ArcControl
from utils.response import get_parameter_dic

class ArcView(APIView):
    def post(self, request, *args, **kwargs):
        params=get_parameter_dic(request)
        return ArcControl.regression(params.get("points"))
