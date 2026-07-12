"""Backend end-to-end exercise, run inside Django (eager Celery, SQLite):

    ingest 60 synthetic docs -> process_document extracts each
    -> auto-review every needs_review/extracted doc (perfect reviewer:
       accepts correct fields, corrects wrong ones from ground truth)
    -> retrain -> champion promoted on frozen holdout
    -> re-ingest 10 more docs, now served by the champion

Run:  python scripts/backend_e2e.py     (from repo root)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from documents.models import InvoiceDocument
from documents.tasks import ping, process_document
from ml.labeling import assign_labels
from ml.synth.generator import generate_dataset
from training.tasks import retrain


def ingest(n: int, seed: int, holdout_every: int = 5) -> list[str]:
    ids = []
    for i, doc in enumerate(generate_dataset(n, seed=seed)):
        assign_labels(doc)
        record = InvoiceDocument.objects.create(
            source="synthetic", doc_json=doc.to_dict(),
            is_holdout=(i % holdout_every == 0),
        )
        process_document.delay(str(record.id))
        ids.append(str(record.id))
    return ids


def auto_review(doc_ids: list[str]) -> int:
    """Simulate a perfect reviewer using the synthetic ground truth."""
    from rest_framework.test import APIClient

    client = APIClient()
    reviewed = 0
    for did in doc_ids:
        record = InvoiceDocument.objects.get(pk=did)
        if record.status not in ("needs_review", "extracted"):
            continue
        truth = record.doc_json["meta"]["truth"]
        ext = record.extractions.first()
        predicted = {f["field"]: f for f in ext.fields}
        corrections = []
        for fname, tval in truth.items():
            pred = predicted.get(fname)
            if pred is not None and str(pred["raw"]).replace(",", "") \
                    .replace("Rs. ", "").replace("₹", "").replace("INR ", "") \
                    .strip() in (str(tval), f"{tval}", f"{float(tval):,.2f}".replace(",", "") if isinstance(tval, (int, float)) else str(tval)):
                corrections.append({"field": fname, "corrected_value": str(pred["raw"]),
                                    "accepted_as_is": True})
            else:
                # Correct (or add) from ground truth, reusing gold bbox.
                gold_bbox = next((a for a in record.doc_json["annotations"]
                                  if a["field"] == fname), None)
                corrections.append({
                    "field": fname, "corrected_value": str(tval),
                    "accepted_as_is": False,
                    "bbox": {k: gold_bbox[k] for k in ("x0", "y0", "x1", "y1", "page")}
                            if gold_bbox else None,
                })
        resp = client.post(f"/api/documents/{did}/review/",
                           {"corrections": corrections, "review_seconds": 45.0},
                           format="json")
        assert resp.status_code == 200, resp.content
        reviewed += 1
    return reviewed


def main() -> None:
    print("== ping task ==")
    assert ping.delay().get() == "pong"
    print("   queue ok (eager mode)")

    print("\n== ingest 60 synthetic docs (baseline engine) ==")
    ids = ingest(60, seed=7)
    by_status = {}
    for d in InvoiceDocument.objects.filter(pk__in=ids):
        by_status[d.status] = by_status.get(d.status, 0) + 1
    print(f"   statuses: {by_status}")

    print("\n== auto-review everything ==")
    n = auto_review(ids)
    print(f"   reviewed {n} docs -> verified")

    print("\n== retrain (champion/challenger) ==")
    print("   " + retrain(triggered_by="e2e"))

    print("\n== ingest 10 more docs, champion should serve ==")
    ids2 = ingest(10, seed=99)
    for d in InvoiceDocument.objects.filter(pk__in=ids2)[:3]:
        ext = d.extractions.first()
        print(f"   {d.status:>13}  avg_conf {ext.avg_confidence:.3f}  "
              f"min_conf {ext.min_confidence:.3f}  {ext.latency_ms:.0f}ms")

    print("\n== monitoring ==")
    from rest_framework.test import APIClient
    client = APIClient()
    m = client.get("/api/monitoring/metrics/").json()
    print(f"   review_rate {m['review_rate']:.2f}  "
          f"avg_confidence {m['avg_confidence']:.3f}  "
          f"avg_review_seconds {m['avg_review_seconds']}")
    d = client.get("/api/monitoring/drift/").json()
    print(f"   drift: {d}")

    print("\nBACKEND E2E PASSED")


if __name__ == "__main__":
    main()
