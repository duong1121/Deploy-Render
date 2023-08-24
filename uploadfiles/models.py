import os
from django.conf import settings
from django.db import models
from django.utils import timezone
# Create your models here.


class Document(models.Model):

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.file)

    def extension(self):
        name, extension = os.path.splitext(self.file.name)
        print(extension)
        if extension == '.pdf':
            return 'pdf'
        if extension == '.docx':
            return 'word'
        if extension == '.jpg' or extension ==  '.png':
            return 'image'
        return 'other'

    def name(self):
        filename = os.path.basename(self.file.name)
        return filename[:20]
