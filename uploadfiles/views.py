import os
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.contrib.auth.forms import UserCreationForm
from django.http.response import JsonResponse, HttpResponse, Http404
from django.views import generic
from .models import Document
from django.views.decorators.csrf import csrf_exempt
from base64 import b64decode, b64encode
import io
from PIL import Image
import numpy as np
from .darknet.darknet import JsonFile
import json
# Create your views here.


@csrf_exempt
def index(request):

       return render(request, 'uploadfiles/index.html')


class UploadView(generic.View):

    def post(self, request, *args, **kwargs):
        file = request.FILES['document'].name
        file_extension = file.split('.')[-1]
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']

        if file_extension.lower() in image_extensions:
            file_extension = 'image'
        else:
            file_extension = 'other'

        image_request = request.FILES['document']
        if file_extension == 'image':
            file_extension = 'image'
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            result = yolo_detect(image_np)
            return JsonResponse({'result': result})
        return JsonResponse({'message': 'not a image'})




class DownloadView(generic.View):

    def get(self, request,  *args, **kwargs):

        data = {'result': 'success'}
        json_data = json.dumps(data, indent=4)

        response = HttpResponse(json_data, content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename="data.json"'
        return response

        return Http404
    

def yolo_detect(original_image):
    return JsonFile(original_image)
