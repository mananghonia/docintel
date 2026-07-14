from django.conf import settings
from django.contrib import admin
from django.http import FileResponse, Http404
from django.urls import include, path, re_path
from django.views import View
from rest_framework.authtoken.views import obtain_auth_token


class SPAView(View):
    """Serve the built React index.html for any non-API route, so client-side
    routes (e.g. /review/<id>) deep-link correctly. WhiteNoise serves the
    hashed assets; this only handles the HTML shell."""

    def get(self, request, *args, **kwargs):
        index = settings.FRONTEND_DIST / "index.html"
        if not index.exists():
            raise Http404("frontend not built (run `npm run build` in frontend/)")
        return FileResponse(open(index, "rb"))


urlpatterns = [
    path("admin/", admin.site.urls),
    # POST {username, password} -> {"token": ...}; send as `Authorization: Token <token>`.
    path("api/auth/token/", obtain_auth_token),
    path("api/documents/", include("documents.urls")),
    path("api/training/", include("training.urls")),
    path("api/monitoring/", include("monitoring.urls")),
    # SPA fallback: anything not matched above returns the React shell.
    re_path(r"^(?!api/|admin/|static/|media/).*$", SPAView.as_view()),
]
