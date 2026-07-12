import numpy as np
from django.db.models import Avg, Count
from rest_framework.decorators import api_view
from rest_framework.response import Response

from documents.models import Correction, Extraction, InvoiceDocument
from monitoring.drift import psi


@api_view(["GET"])
def metrics(request):
    """Operational dashboard numbers: volumes, review rate, review effort."""
    docs = InvoiceDocument.objects.values("status").annotate(n=Count("id"))
    n_total = InvoiceDocument.objects.count()
    n_review = InvoiceDocument.objects.filter(
        status__in=["needs_review", "verified"]).count()

    corrections = Correction.objects.aggregate(
        n=Count("id"), avg_seconds=Avg("review_seconds"))
    n_changed = Correction.objects.filter(accepted_as_is=False).count()

    exts = Extraction.objects.aggregate(
        avg_conf=Avg("avg_confidence"), avg_latency=Avg("latency_ms"))

    return Response({
        "documents_by_status": {d["status"]: d["n"] for d in docs},
        "review_rate": n_review / n_total if n_total else None,
        "corrections_total": corrections["n"],
        "corrections_changed": n_changed,
        "field_acceptance_rate": (1 - n_changed / corrections["n"])
                                  if corrections["n"] else None,
        "avg_review_seconds": corrections["avg_seconds"],
        "avg_confidence": exts["avg_conf"],
        "avg_latency_ms": exts["avg_latency"],
    })


@api_view(["GET"])
def drift(request):
    """PSI of the confidence distribution: first half of extractions as
    reference vs the most recent quarter as current."""
    confs = list(Extraction.objects.order_by("created_at")
                 .values_list("avg_confidence", flat=True))
    if len(confs) < 40:
        return Response({"psi": None,
                         "detail": f"need >= 40 extractions, have {len(confs)}"})
    arr = np.array(confs)
    ref, cur = arr[: len(arr) // 2], arr[-len(arr) // 4:]
    value = psi(ref, cur)
    status = "stable" if value < 0.1 else ("drifting" if value < 0.25 else "alert")
    return Response({"psi": round(value, 4), "status": status,
                     "reference_n": len(ref), "current_n": len(cur)})
