from django.db import models
from django.contrib.auth.models import User


class CompanyInfo(models.Model):

    name_of_company = models.CharField(max_length=30)
    email_of_company = models.CharField(max_length=30, unique=True)
    password_of_company=models.CharField(max_length=20)
    activation_code=models.CharField(max_length=30)
    number_of_camera = models.IntegerField(default=0)

    def __str__(self):
        return self.email_of_company


class CamerasInfo(models.Model):
    name_of_camera = models.CharField(max_length=30)
    ip_of_camera = models.CharField(max_length=50)
    company_info_id = models.ForeignKey(CompanyInfo, on_delete=models.CASCADE)
    is_running = models.BooleanField()

    def __str__(self):
        return self.name_of_camera