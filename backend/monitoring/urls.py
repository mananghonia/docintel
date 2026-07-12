from django.urls import path

from monitoring.views import drift, metrics

urlpatterns = [
    path("metrics/", metrics),
    path("drift/", drift),
]
