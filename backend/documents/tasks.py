import time

from celery import shared_task
from django.conf import settings


@shared_task
def ping() -> str:
    """Week-1 gate: prove the queue works."""
    return "pong"


@shared_task
def process_document(doc_id: str) -> str:
    """OCR (if needed) -> predict -> postprocess -> Extraction row.

    Documents whose every field clears CONFIDENCE_REVIEW_THRESHOLD are
    auto-accepted; the rest go to the review queue.
    """
    from documents.inference import build_fields, predict_document
    from documents.models import Extraction, InvoiceDocument
    from ml.labeling import Document

    record = InvoiceDocument.objects.get(pk=doc_id)
    record.status = InvoiceDocument.Status.PROCESSING
    record.save(update_fields=["status"])

    try:
        if record.doc_json:  # synthetic / pre-tokenised
            doc = Document.from_dict(record.doc_json)
        elif record.file:
            doc = _ocr_file(record)
            record.doc_json = doc.to_dict()
        else:
            raise ValueError("Document has neither file nor doc_json")

        t0 = time.perf_counter()
        tags, probs, engine = predict_document(doc)
        latency_ms = (time.perf_counter() - t0) * 1000
        fields = build_fields(doc, tags, probs)

        confs = [f["confidence"] for f in fields] or [0.0]
        needs_review = (min(confs) < settings.CONFIDENCE_REVIEW_THRESHOLD
                        or not fields)
        Extraction.objects.create(
            document=record,
            fields=fields,
            token_tags=tags,
            avg_confidence=sum(confs) / len(confs),
            min_confidence=min(confs),
            needs_review=needs_review,
            latency_ms=latency_ms,
        )
        record.status = (InvoiceDocument.Status.NEEDS_REVIEW if needs_review
                         else InvoiceDocument.Status.EXTRACTED)
        record.error = ""
        record.save()
        return f"{engine}: {len(fields)} fields, review={needs_review}"
    except Exception as e:  # noqa: BLE001 - persist the failure for the UI
        record.status = InvoiceDocument.Status.FAILED
        record.error = str(e)
        record.save(update_fields=["status", "error"])
        raise


def _ocr_file(record):
    from ml.ocr import ocr_image, ocr_pdf

    path = record.file.path
    if path.lower().endswith(".pdf"):
        return ocr_pdf(path, doc_id=str(record.id))
    from PIL import Image

    return ocr_image(Image.open(path), doc_id=str(record.id))
