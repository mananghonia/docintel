"""Register the baked champion.joblib as ModelVersion v1 so the dashboard shows
a champion on a fresh deployment (the build bakes the model file but not a DB
row). No-op if a ModelVersion already exists or no model file is present.
"""

from pathlib import Path

import joblib
from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Register the baked champion model as ModelVersion v1."

    def handle(self, *args, **opts):
        from training.models import ModelVersion

        path = Path(settings.MODEL_DIR) / "champion.joblib"
        if ModelVersion.objects.exists():
            self.stdout.write("a ModelVersion already exists — skipping")
            return
        if not path.exists():
            self.stdout.write("no champion.joblib — skipping")
            return

        bundle = joblib.load(path)
        model = bundle["model"]
        temp = getattr(bundle.get("scaler"), "temperature", 1.0)

        # Quick holdout F1 on fresh synthetic invoices so the dashboard shows a
        # real number rather than a placeholder.
        from ml.evaluate import evaluate_field_extraction
        from ml.features import featurize_document
        from ml.labeling import ID2TAG, assign_labels
        from ml.synth.generator import generate_dataset

        docs = [assign_labels(d) for d in generate_dataset(30, seed=999, hard=True)]
        tag_lists = [[ID2TAG[int(p)] for p in model.predict(featurize_document(d))]
                     for d in docs]
        f1 = evaluate_field_extraction(docs, tag_lists).macro_f1()

        ModelVersion.objects.create(
            version=1, artifact_path=str(path), is_champion=True,
            n_training_docs=400,
            metrics={"field_macro_f1": f1, "temperature": temp,
                     "note": "baked at build from broadened synthetic data"})
        self.stdout.write(self.style.SUCCESS(
            f"registered baked champion v1 (holdout field-F1 {f1:.3f})"))
