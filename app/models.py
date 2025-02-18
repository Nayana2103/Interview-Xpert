from django.db import models

class regtable(models.Model):
    name=models.CharField(max_length=150) 
    email=models.CharField(max_length=150)
    password=models.CharField(max_length=150) 
    phone=models.CharField(max_length=150)
    resume=models.CharField(max_length=150)


class InterviewAnalysis(models.Model):
    user_id=models.CharField(max_length=150) 
    calm=models.CharField(max_length=150) 
    excitement=models.CharField(max_length=150) 
    stress=models.CharField(max_length=150) 
    response_score=models.CharField(max_length=150) 
    confidence_score=models.CharField(max_length=150)
    resume_analysis=models.CharField(max_length=500)

 