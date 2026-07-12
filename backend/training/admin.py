from django.contrib import admin

from training.models import ModelVersion, TrainingRun

admin.site.register(ModelVersion)
admin.site.register(TrainingRun)
