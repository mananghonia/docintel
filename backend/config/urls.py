from django.contrib import admin
from django.urls import include, path
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    path("admin/", admin.site.urls),
    # POST {username, password} -> {"token": ...}; send as `Authorization: Token <token>`.
    path("api/auth/token/", obtain_auth_token),
    path("api/documents/", include("documents.urls")),
    path("api/training/", include("training.urls")),
    path("api/monitoring/", include("monitoring.urls")),
]
