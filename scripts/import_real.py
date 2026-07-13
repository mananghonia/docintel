"""Import real labeled documents into DocIntel's Document format.

Two sources:

1. CORD v2 (receipts, HuggingFace `naver-clova-ix/cord-v2`):
       pip install datasets pillow
       python scripts/import_real.py cord --split train --limit 150
   Maps CORD line categories onto our schema where they overlap
   (total_amount, subtotal, tax_amount). Everything unmapped stays O —
   partial labels are still useful: amounts are the fields the arithmetic
   rule needs.

2. Your own invoices — no script needed: upload through the app, correct in
   the review UI, and the verified doc_json IS a labeled Document. Export:
       python scripts/import_real.py export-verified

Output: one JSON per document in data/labeled/, loadable by
`scripts/train.py --real data/labeled`.
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
    ap.add_argument("command", choices=["cord", "export-verified"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()
    if args.command == "cord":
        import_cord(args.split, args.limit)
    else:
        export_verified()
