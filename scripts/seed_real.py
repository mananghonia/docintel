"""Load labeled real documents (data/labeled/*.json) into the app as verified
documents and run one retrain through the champion/challenger gate.

This is the "prove the loop works on real data" demonstration: the current
champion (trained on synthetic invoices) is scored against a challenger
trained on real documents, on a REAL held-out set neither has trained on.

    python scripts/seed_real.py            # load data/labeled + retrain
    python scripts/seed_real.py --dir X    # load a different directory

Idempotent: re-running replaces the previously seeded real docs.
"""

import argparse
import json
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
from ml.evaluate import evaluate_field_extraction  # noqa: E402
from ml.features import featurize_document  # noqa: E402
from ml.labeling import ID2TAG, Document  # noqa: E402

SOURCE = "real-cord"


def load(dir_path: Path, holdout_every: int, real_only: bool) -> tuple[int, int]:
    InvoiceDocument.objects.filter(source=SOURCE).delete()  # idempotent reseed
    if real_only:
        # Remove synthetic test-fixture docs so training AND the frozen holdout
        # are real invoices — otherwise a large synthetic holdout dominates the
        # gate and hides the real-data improvement (a real lesson: the holdout
        # must reflect the distribution you actually care about).
        n_removed = InvoiceDocument.objects.filter(
            status=InvoiceDocument.Status.VERIFIED, source="synthetic").delete()[0]
        if n_removed:
            print(f"   removed {n_removed} synthetic test-fixture docs "
                  f"(real-only retrain)")
    files = sorted(dir_path.glob("*.json"))
    if not files:
        sys.exit(f"no *.json in {dir_path} — run scripts/import_real.py cord first")
    n_hold = 0
    for i, p in enumerate(files):
        doc = Document.from_dict(json.loads(p.read_text(encoding="utf-8")))
        is_holdout = (i % holdout_every == 0)
        n_hold += is_holdout
        InvoiceDocument.objects.create(
            source=SOURCE, status=InvoiceDocument.Status.VERIFIED,
            doc_json=doc.to_dict(), is_holdout=is_holdout,
            verified_at=timezone.now())
    return len(files), n_hold


def champion_f1_on_real_holdout() -> float | None:
    """Score the CURRENT champion on the real held-out docs only."""
    import joblib
    from django.conf import settings
    from training.models import ModelVersion

    champ = ModelVersion.objects.filter(is_champion=True).first()
    path = settings.MODEL_DIR / "champion.joblib"
    if not champ or not path.exists():
        return None
    model = joblib.load(path)["model"]
    docs = [Document.from_dict(r.doc_json) for r in
            InvoiceDocument.objects.filter(source=SOURCE, is_holdout=True)]
    if not docs:
        return None
    tag_lists = [[ID2TAG[int(p)] for p in model.predict(featurize_document(d))]
                 for d in docs]
    return evaluate_field_extraction(docs, tag_lists).macro_f1()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "labeled"))
    ap.add_argument("--holdout-every", type=int, default=4)
    ap.add_argument("--keep-synthetic", action="store_true",
                    help="keep synthetic verified docs in the training pool")
    args = ap.parse_args()

    print(f"== loading real docs from {args.dir} ==")
    n, n_hold = load(Path(args.dir), args.holdout_every,
                     real_only=not args.keep_synthetic)
    print(f"   {n} real docs loaded as VERIFIED ({n_hold} frozen holdout, "
          f"{n - n_hold} train)")

    before = champion_f1_on_real_holdout()
    print(f"\n== current champion on real holdout (BEFORE): "
          f"{'n/a (no champion)' if before is None else f'{before:.3f}'} ==")

    print("\n== retrain through champion/challenger gate ==")
    from training.tasks import retrain
    print("   " + retrain(triggered_by="real-data"))

    after = champion_f1_on_real_holdout()
    print(f"\n== champion on real holdout (AFTER): "
          f"{'n/a' if after is None else f'{after:.3f}'} ==")
    if before is not None and after is not None:
        print(f"   real-data field macro-F1: {before:.3f} -> {after:.3f} "
              f"({after - before:+.3f})")


if __name__ == "__main__":
    main()
