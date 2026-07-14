"""Set up a strong, consistent demo champion.

Seeds the app with a broadened synthetic training set (Indian GST *and* US/EU
invoices) plus the real CORD documents, then retrains through the
champion/challenger gate so the promoted model handles realistic invoices —
not only Indian tax invoices — and the dashboard's ModelVersion is consistent.

    python scripts/seed_demo.py [--n 250]

Then reprocesses any real uploads so their extractions reflect the new model.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from django.utils import timezone  # noqa: E402

from documents.models import InvoiceDocument  # noqa: E402
from ml.labeling import assign_labels  # noqa: E402
from ml.synth.generator import generate_document  # noqa: E402

SOURCE = "synthetic-demo"


def seed_synthetic(n: int, holdout_every: int) -> tuple[int, int]:
    InvoiceDocument.objects.filter(source=SOURCE).delete()
    # Remove stale plain-synthetic fixtures so the training pool is the new
    # broadened set + real data.
    InvoiceDocument.objects.filter(source="synthetic").delete()
    n_hold = 0
    for i in range(n):
        # Mix clean and hard (OCR-noised) so the model is robust but still
        # sharp on cleanly-rendered invoices like a downloaded PDF.
        doc = assign_labels(generate_document(seed=10_000 + i, hard=(i % 2 == 0)))
        is_holdout = (i % holdout_every == 0)
        n_hold += is_holdout
        InvoiceDocument.objects.create(
            source=SOURCE, status=InvoiceDocument.Status.VERIFIED,
            doc_json=doc.to_dict(), is_holdout=is_holdout,
            verified_at=timezone.now())
    return n, n_hold


def reprocess_uploads() -> None:
    from documents.tasks import process_document

    ups = InvoiceDocument.objects.filter(source="upload")
    for d in ups:
        try:
            process_document(str(d.id))
        except Exception as e:  # noqa: BLE001
            print(f"   reprocess {str(d.id)[:8]} failed: {e}")
    print(f"   reprocessed {ups.count()} uploaded documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--holdout-every", type=int, default=5)
    args = ap.parse_args()

    print(f"== seeding {args.n} broadened synthetic invoices (IN + US/EU) ==")
    n, n_hold = seed_synthetic(args.n, args.holdout_every)
    n_real = InvoiceDocument.objects.filter(
        source="real-cord", status=InvoiceDocument.Status.VERIFIED).count()
    print(f"   {n} synthetic ({n_hold} holdout) + {n_real} real docs verified")

    print("\n== retrain through champion/challenger gate ==")
    from training.tasks import retrain
    print("   " + retrain(triggered_by="demo-seed"))

    print("\n== reprocess real uploads with the new champion ==")
    reprocess_uploads()

    # Show the effect on any uploaded real invoices.
    print("\n== extractions on your uploads now ==")
    for d in InvoiceDocument.objects.filter(source="upload").order_by("-created_at")[:4]:
        ext = d.extractions.first()
        fname = (d.file.name.split("/")[-1] if d.file else "?")[:34]
        nf = len(ext.fields) if ext else 0
        print(f"   {fname:<36} {d.status:<13} {nf} fields")


if __name__ == "__main__":
    main()
