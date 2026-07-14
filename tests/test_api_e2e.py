"""End-to-end HTTP flow through the real API (the wiring the unit tests skip).

Drives the product loop with Django's test client on a throwaway test DB, with
Celery eager (no broker), so ingest actually runs OCR-free extraction inline:

    ingest_synthetic -> extraction created -> export -> review -> verified

Marked django_db so pytest-django builds the test database and runs migrations.
"""

import pytest
from rest_framework.test import APIClient


@pytest.fixture
def client():
    return APIClient()


@pytest.mark.django_db
def test_ingest_extract_export_review_flow(client):
    # 1. Ingest synthetic docs — eager Celery runs extraction inline.
    r = client.post("/api/documents/ingest_synthetic/", {"n": 6, "holdout_every": 3},
                    format="json")
    assert r.status_code == 201
    ids = r.json()["created"]
    assert len(ids) == 6

    # 2. Every doc reached a terminal extraction state.
    from documents.models import InvoiceDocument
    for did in ids:
        d = InvoiceDocument.objects.get(pk=did)
        assert d.status in ("extracted", "needs_review")
        assert d.extractions.exists()

    # 3. Export returns structured fields for a document.
    exp = client.get(f"/api/documents/{ids[0]}/export/")
    assert exp.status_code == 200
    body = exp.json()
    assert body["document_id"] == ids[0]
    assert isinstance(body["fields"], list) and len(body["fields"]) > 0
    assert {"field", "value", "confidence"} <= set(body["fields"][0])

    # 4. Review a document with the synthetic ground truth -> verified.
    d = InvoiceDocument.objects.get(pk=ids[0])
    truth = d.doc_json["meta"]["truth"]
    ext = d.extractions.first()
    predicted = {f["field"]: f for f in ext.fields}
    corrections = []
    for fname, tval in truth.items():
        gold = next((a for a in d.doc_json["annotations"] if a["field"] == fname), None)
        corrections.append({
            "field": fname, "corrected_value": str(tval), "accepted_as_is": False,
            "bbox": {k: gold[k] for k in ("x0", "y0", "x1", "y1", "page")} if gold else None,
        })
    rr = client.post(f"/api/documents/{ids[0]}/review/",
                     {"corrections": corrections, "review_seconds": 30.0}, format="json")
    assert rr.status_code == 200
    d.refresh_from_db()
    assert d.status == "verified"
    assert d.verified_at is not None
    assert ext.corrections.count() == len(corrections)


@pytest.mark.django_db
def test_upload_rejects_bad_extension(client):
    from io import BytesIO
    bad = BytesIO(b"not a document"); bad.name = "malware.exe"
    r = client.post("/api/documents/upload/", {"file": bad}, format="multipart")
    assert r.status_code == 400


@pytest.mark.django_db
def test_review_queue_and_metrics(client):
    client.post("/api/documents/ingest_synthetic/", {"n": 5, "holdout_every": 0},
                format="json")
    q = client.get("/api/documents/review_queue/")
    assert q.status_code == 200
    m = client.get("/api/monitoring/metrics/")
    assert m.status_code == 200
    assert "documents_by_status" in m.json()
