import logging
from utils.response import JsonResponse
from rest_framework import status
LOG = logging.getLogger(__name__)

class LineControl(object):
    @classmethod
    def regression(cls, data):
        back = {"a":"0", "b":"0"}
        for point in data:
            print("x: {} --- y: {} ".format(point.get("x"), point["y"]))
        return JsonResponse(data=back, code=status.HTTP_200_OK, desc='get house success') 