import os
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.urls import reverse_lazy
from django.contrib.auth.forms import UserCreationForm
from django.http.response import JsonResponse, HttpResponse, Http404
from django.views import generic
from .models import Document
# Create your views here.


@login_required
def index(request):
        documents = Document.objects.filter(user=request.user)
        return render(request, 'uploadfiles/index.html', {'documents': documents})


class UploadView(LoginRequiredMixin, generic.View):

    def post(self, request, *args, **kwargs):

        user = request.user
        file = request.FILES['document']
        Document.objects.create(user=user, file=file)
        return JsonResponse({'message': 'success'})


class DownloadView(LoginRequiredMixin, generic.View):

    def get(self, request, id,  *args, **kwargs):

        document_obj = Document.objects.get(id=id)
        if request.user == document_obj.user:
            file = document_obj.file.path
            response = HttpResponse(open(file, 'rb').read(), content_type="application/default")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file)

            return response

        return Http404


class UserRegisterView(generic.CreateView):

    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'uploadfiles/register.html'
