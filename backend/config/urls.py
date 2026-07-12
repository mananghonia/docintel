from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/documents/", include("documents.urls")),
    path("api/training/", include("training.urls")),
    path("api/monitoring/", include("monitoring.urls")),
]
