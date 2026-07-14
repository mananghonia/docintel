"""Seed a fresh deployment with a few processed synthetic documents so the
dashboard and review queue aren't empty on first load. No-op if any documents
already exist.

    python manage.py bootstrap_demo --n 12
"""

from django.core.management.base import BaseCommand

from documents.models import InvoiceDocument
from documents.tasks import process_document
from ml.labeling import assign_labels
from ml.synth.generator import generate_dataset


class Command(BaseCommand):
    help = "Seed synthetic documents on an empty deployment."

    def add_arguments(self, parser):
        parser.add_argument("--n", type=int, default=12)

    def handle(self, *args, **opts):
        if InvoiceDocument.objects.exists():
            self.stdout.write("documents already present — skipping demo seed")
            return
        n = opts["n"]
        for i, doc in enumerate(generate_dataset(n, seed=20_000)):
            assign_labels(doc)
            rec = InvoiceDocument.objects.create(
                source="synthetic", doc_json=doc.to_dict(),
                is_holdout=(i % 5 == 0))
            process_document(str(rec.id))  # eager in the deploy image
        self.stdout.write(self.style.SUCCESS(f"seeded {n} demo documents"))
