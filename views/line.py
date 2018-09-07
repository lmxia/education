from rest_framework.views import APIView
from controls.line import LineControl
from utils.response import get_parameter_dic
# def get_parameter_dic(request, *args, **kwargs):
#     if isinstance(request, Request) == False:
#         return {}

#     query_params = request.query_params
#     if isinstance(query_params, QueryDict):
#         query_params = query_params.dict()
#     result_data = request.data
#     if isinstance(result_data, QueryDict):
#         result_data = result_data.dict()

#     if query_params != {}:
#         return query_params
#     else:
#         return result_data

class LineView(APIView):
    def post(self, request, *args, **kwargs):
        params=get_parameter_dic(request)
        return LineControl.regression(params.get("points"))
