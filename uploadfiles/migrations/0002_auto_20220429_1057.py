# Generated by Django 3.2.13 on 2022-04-29 05:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('uploadfiles', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='document',
            name='description',
        ),
        migrations.AlterField(
            model_name='document',
            name='uploaded_at',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
