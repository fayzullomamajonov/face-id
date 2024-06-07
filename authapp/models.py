from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    age = models.IntegerField()
    regions = models.CharField(max_length=255)
    photo = models.ImageField(upload_to='photos/')
    
    def __str__(self):
        return self.first_name + ' ' + self.last_name
