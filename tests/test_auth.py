"""Authentication gate (#7): open in dev, enforced when REQUIRE_AUTH is on."""

import pytest
from django.test import override_settings
from rest_framework.test import APIClient


@pytest.mark.django_db
def test_open_in_dev_default():
    # Default test settings have DEBUG on -> AllowAny; anonymous calls work.
    r = APIClient().get("/api/monitoring/metrics/")
    assert r.status_code == 200


@pytest.mark.django_db
def test_enforced_when_required(monkeypatch):
    # Simulate a REQUIRE_AUTH=1 deployment by forcing IsAuthenticated on the
    # view (reliable in-process; avoids DRF's global-settings cache in pytest).
    from rest_framework.permissions import IsAuthenticated
    from documents.views import DocumentViewSet
    monkeypatch.setattr(DocumentViewSet, "permission_classes", [IsAuthenticated])

    # Anonymous is refused...
    assert APIClient().get("/api/documents/").status_code in (401, 403)

    # ...and a token grants access.
    from django.contrib.auth import get_user_model
    from rest_framework.authtoken.models import Token
    user = get_user_model().objects.create_user("t", password="x")
    token = Token.objects.create(user=user)
    auth = APIClient()
    auth.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    assert auth.get("/api/documents/").status_code == 200


def test_require_auth_selects_permission():
    # The settings wiring: REQUIRE_AUTH -> IsAuthenticated, else AllowAny.
    from django.conf import settings
    perms = settings.REST_FRAMEWORK["DEFAULT_PERMISSION_CLASSES"]
    expected = ("IsAuthenticated" if settings.REQUIRE_AUTH else "AllowAny")
    assert expected in perms[0]


@pytest.mark.django_db
def test_obtain_token_endpoint():
    from django.contrib.auth import get_user_model
    get_user_model().objects.create_user("bob", password="hunter2")
    r = APIClient().post("/api/auth/token/", {"username": "bob", "password": "hunter2"})
    assert r.status_code == 200 and "token" in r.json()
