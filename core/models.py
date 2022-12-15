from django.db import models
from django.contrib.auth import get_user_model
import uuid
from datetime import datetime

User = get_user_model()

class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    id_datalogger = models.IntegerField()
    location = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.user.username

class DataListrik(models.Model):
    class Meta:
        verbose_name_plural = "ID_Datalogger"

    datetime = models.CharField(max_length=10) # models.DateTimeField()
    voltage = models.CharField(max_length=10)
    current = models.CharField(max_length=10)
    power = models.CharField(max_length=10)
    frequency = models.CharField(max_length=10)
    power_factor = models.CharField(max_length=10)
    current_ch1 = models.CharField(max_length=10)
    current_ch2 = models.CharField(max_length=10)
    current_ch3 = models.CharField(max_length=10)
    
    def __str__(self):
        return self.datetime

class TestSimpan(models.Model):
    class Meta:
        verbose_name_plural = "Test_Simpan"

    datetime = models.CharField(max_length=20) # models.DateTimeField()
    data1 = models.CharField(max_length=10)
    data2 = models.CharField(max_length=10)
    
    def __str__(self):
        return self.datetime



# notifikasi, datalogger_issue