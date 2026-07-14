"""Import real labeled documents into DocIntel's Document format.

Three sources:

1. DocILE (real INVOICES, the gold-standard dataset for this task — maps onto
   the full 12-field schema, not just amounts):
       # one-time: get a token at https://docile.rossum.ai and download it
       pip install docile-benchmark
       ./download_dataset.sh TOKEN annotated-trainval data/docile --unzip
       python scripts/import_real.py docile --docile-path data/docile --limit 300

2. CORD v2 (receipts, HuggingFace `naver-clova-ix/cord-v2`):
       pip install datasets pillow
       python scripts/import_real.py cord --split train --limit 150
   Only amount fields overlap our schema.

3. Your own invoices — no script needed: upload through the app, correct in
   the review UI, and the verified doc_json IS a labeled Document. Export:
       python scripts/import_real.py export-verified

Output: one JSON per document in data/labeled/, loadable by
`scripts/train.py --real data/labeled`, `scripts/seed_real.py`, and
`scripts/seed_demo.py`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "labeled"

CORD_CATEGORY_MAP = {
    "total.total_price": "total_amount",
    "sub_total.subtotal_price": "subtotal",
    "sub_total.tax_price": "tax_amount",
}

# DocILE KILE field types -> our schema. Multiple DocILE names can map to one
# of our fields (the dataset splits, e.g., several tax/amount granularities);
# the first match on a page wins. Unmapped field types stay O — partial labels
# are still useful. Field-type names follow the DocILE benchmark (55 classes).
DOCILE_FIELD_MAP = {
    "document_id": "invoice_number",
    "date_issue": "invoice_date",
    "date_due": "due_date",
    "vendor_name": "vendor_name",
    "vendor_tax_id": "vendor_gstin",
    "vendor_registration_id": "vendor_gstin",
    "customer_name": "buyer_name",
    "customer_billing_name": "buyer_name",
    "customer_tax_id": "buyer_gstin",
    "customer_registration_id": "buyer_gstin",
    "amount_untaxed": "subtotal",
    "amount_total_base": "subtotal",
    "amount_total_tax": "tax_amount",
    "tax_detail_tax": "tax_amount",
    "amount_total_gross": "total_amount",
    "amount_due": "total_amount",
    "currency_code_amount_due": "currency",
    "currency_code_amount_total": "currency",
    "order_id": "po_number",
    "payment_reference": "po_number",
}

# DocILE bboxes are relative [0,1]; render onto a nominal A4 canvas so tokens
# and annotations share one coordinate space (features normalise by page size).
DOCILE_PAGE_W, DOCILE_PAGE_H = 1240, 1754


def import_cord(split: str, limit: int) -> None:
    from datasets import load_dataset

    from ml.labeling import Document, FieldAnnotation, Token, assign_labels

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)

    n_written = 0
    for i, sample in enumerate(ds):
        if n_written >= limit:
            break
        gt = json.loads(sample["ground_truth"])
        lines = gt.get("valid_line", [])
        if not lines:
            continue
        img = sample["image"]

        tokens, annotations = [], []
        for line in lines:
            field = CORD_CATEGORY_MAP.get(line.get("category", ""))
            words = line.get("words", [])
            span = []
            for w in words:
                q = w["quad"]
                x0, y0 = min(q["x1"], q["x4"]), min(q["y1"], q["y2"])
                x1, y1 = max(q["x2"], q["x3"]), max(q["y3"], q["y4"])
                tokens.append(Token(text=w["text"], x0=x0, y0=y0, x1=x1, y1=y1))
                span.append((x0, y0, x1, y1))
            if field and span:
                annotations.append(FieldAnnotation(
                    field=field,
                    value=" ".join(w["text"] for w in words),
                    x0=min(s[0] for s in span) - 2, y0=min(s[1] for s in span) - 2,
                    x1=max(s[2] for s in span) + 2, y1=max(s[3] for s in span) + 2,
                ))

        doc = Document(
            doc_id=f"cord-{split}-{i:04d}",
            tokens=tokens,
            page_width=img.width,
            page_height=img.height,
            annotations=annotations,
            meta={"source": "cord-v2",
                  "truth": {a.field: a.value for a in annotations}},
        )
        assign_labels(doc)
        (OUT_DIR / f"{doc.doc_id}.json").write_text(
            json.dumps(doc.to_dict()), encoding="utf-8")
        n_written += 1

    print(f"wrote {n_written} documents to {OUT_DIR}")
    print("NOTE: CORD is receipts, not invoices — only amount fields map. "
          "Evaluate per-field, not macro over the full schema.")


def _bbox_tuple(bbox):
    """DocILE BBox -> (left, top, right, bottom) relative floats, tolerant of
    a few API shapes across docile-benchmark versions."""
    for attr in ("to_tuple", "as_tuple"):
        fn = getattr(bbox, attr, None)
        if callable(fn):
            return tuple(fn())
    if all(hasattr(bbox, a) for a in ("left", "top", "right", "bottom")):
        return (bbox.left, bbox.top, bbox.right, bbox.bottom)
    return tuple(bbox)  # already a sequence


def import_docile(docile_path: str, split: str, limit: int) -> None:
    """Load DocILE KILE annotations into DocIntel Documents (full schema).

    Uses the precomputed OCR shipped with the annotated set, so no page images
    or PDFs are needed. Field types are mapped via DOCILE_FIELD_MAP.
    """
    try:
        from docile.dataset import Dataset
    except ImportError:
        sys.exit("pip install docile-benchmark first (see the module docstring "
                 "for the download steps and token).")

    from ml.labeling import Document, FieldAnnotation, Token, assign_labels

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(split, docile_path)
    W, H = DOCILE_PAGE_W, DOCILE_PAGE_H

    n_written = 0
    for doc in dataset:
        if n_written >= limit:
            break
        try:
            n_pages = doc.page_count
            kile_fields = list(doc.annotation.fields)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {getattr(doc, 'docid', '?')}: {e}")
            continue

        tokens, annotations, truth = [], [], {}
        for page in range(n_pages):
            for w in doc.ocr.get_all_words(page, snapped=True):
                text = (w.text or "").strip()
                if not text:
                    continue
                l, t, r, b = _bbox_tuple(w.bbox)
                tokens.append(Token(text=text, x0=l * W, y0=t * H,
                                    x1=r * W, y1=b * H, page=page))
            for f in kile_fields:
                if getattr(f, "page", 0) != page:
                    continue
                field = DOCILE_FIELD_MAP.get(f.fieldtype)
                if field is None or field in truth:
                    continue  # unmapped, or first-span-wins for this field
                l, t, r, b = _bbox_tuple(f.bbox)
                annotations.append(FieldAnnotation(
                    field=field, value=(f.text or "").strip(), page=page,
                    x0=l * W - 2, y0=t * H - 2, x1=r * W + 2, y1=b * H + 2))
                truth[field] = (f.text or "").strip()

        if not annotations:
            continue
        out = Document(doc_id=f"docile-{doc.docid}", tokens=tokens,
                       page_width=W, page_height=H, annotations=annotations,
                       meta={"source": "docile", "truth": truth})
        assign_labels(out)
        (OUT_DIR / f"{out.doc_id}.json").write_text(
            json.dumps(out.to_dict()), encoding="utf-8")
        n_written += 1

    print(f"wrote {n_written} DocILE documents to {OUT_DIR}")
    print("These map onto the full 12-field schema. Next:")
    print("  python scripts/seed_real.py           # retrain on real invoices")


def export_verified() -> None:
    """Export human-verified documents from the app DB as labeled JSONs."""
    import os

    sys.path.insert(0, str(ROOT / "backend"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()
    from documents.models import InvoiceDocument

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qs = InvoiceDocument.objects.filter(
        status=InvoiceDocument.Status.VERIFIED).exclude(source="synthetic")
    n = 0
    for rec in qs:
        (OUT_DIR / f"verified-{rec.id}.json").write_text(
            json.dumps(rec.doc_json), encoding="utf-8")
        n += 1
    print(f"exported {n} verified real documents to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["docile", "cord", "export-verified"])
    ap.add_argument("--split", default="train",
                    help="cord: train/test; docile: annotated-trainval / test")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--docile-path", default=str(ROOT / "data" / "docile"),
                    help="path passed to download_dataset.sh")
    args = ap.parse_args()
    if args.command == "docile":
        split = "annotated-trainval" if args.split == "train" else args.split
        import_docile(args.docile_path, split, args.limit)
    elif args.command == "cord":
        import_cord(args.split, args.limit)
    else:
        export_verified()
