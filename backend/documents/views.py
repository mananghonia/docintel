import os

from django.conf import settings
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from documents.models import Correction, InvoiceDocument
from documents.serializers import (DocumentDetailSerializer,
                                   DocumentListSerializer, ReviewSerializer)
from documents.tasks import process_document


class DocumentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = InvoiceDocument.objects.all()

    def get_serializer_class(self):
        return (DocumentDetailSerializer if self.action == "retrieve"
                else DocumentListSerializer)

    def get_queryset(self):
        qs = super().get_queryset()
        if s := self.request.query_params.get("status"):
            qs = qs.filter(status=s)
        return qs

    @action(detail=False, methods=["post"])
    def upload(self, request):
        f = request.FILES.get("file")
        if not f:
            return Response({"detail": "multipart 'file' required"},
                            status=status.HTTP_400_BAD_REQUEST)
        # Guard the ingestion surface: an unbounded upload of an arbitrary
        # file type is a trivial DoS and a route to feeding junk to the OCR.
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
            return Response(
                {"detail": f"unsupported file type {ext!r}; allowed: "
                           f"{sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}"},
                status=status.HTTP_400_BAD_REQUEST)
        if f.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            return Response(
                {"detail": f"file too large ({f.size / 1e6:.1f} MB); "
                           f"limit is {settings.MAX_UPLOAD_MB} MB"},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        doc = InvoiceDocument.objects.create(file=f, source="upload")
        process_document.delay(str(doc.id))
        return Response(DocumentListSerializer(doc).data,
                        status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["post"])
    def ingest_synthetic(self, request):
        """Generate n synthetic invoices and run them through the pipeline —
        lets the whole system be demoed with zero real files or OCR."""
        from ml.labeling import assign_labels
        from ml.synth.generator import generate_dataset

        n = min(int(request.data.get("n", 10)), 200)
        holdout_every = int(request.data.get("holdout_every", 5))
        created = []
        for i, doc in enumerate(generate_dataset(n, seed=int(timezone.now().timestamp()))):
            assign_labels(doc)
            record = InvoiceDocument.objects.create(
                source="synthetic",
                doc_json=doc.to_dict(),
                is_holdout=(holdout_every > 0 and i % holdout_every == 0),
            )
            process_document.delay(str(record.id))
            created.append(str(record.id))
        return Response({"created": created}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["get"])
    def review_queue(self, request):
        """Docs needing review, most informative first (lowest min confidence
        as a proxy; ml.active_learning ranks batches during retraining)."""
        qs = (self.get_queryset()
              .filter(status=InvoiceDocument.Status.NEEDS_REVIEW)
              .order_by("extractions__min_confidence")[:50])
        return Response(DocumentListSerializer(qs, many=True).data)

    @action(detail=True, methods=["get"])
    def export(self, request, pk=None):
        """Structured extraction as clean JSON — the machine-readable output
        of the pipeline (what a downstream system consumes)."""
        record = self.get_object()
        ext = record.extractions.first()
        if ext is None:
            return Response({"detail": "no extraction yet"},
                            status=status.HTTP_404_NOT_FOUND)
        return Response({
            "document_id": str(record.id),
            "status": record.status,
            "model_confidence": {"avg": ext.avg_confidence, "min": ext.min_confidence},
            "fields": [{"field": f["field"], "value": f["value"],
                        "confidence": f["confidence"], "flags": f.get("flags", [])}
                       for f in ext.fields],
        })

    @action(detail=True, methods=["post"])
    def review(self, request, pk=None):
        """Apply a reviewer's corrections: store deltas, rebuild gold
        annotations in doc_json, mark the document verified."""
        record = self.get_object()
        extraction = record.extractions.first()
        if extraction is None:
            return Response({"detail": "no extraction to review"},
                            status=status.HTTP_409_CONFLICT)
        ser = ReviewSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        predicted = {f["field"]: f for f in extraction.fields}
        review_seconds = ser.validated_data.get("review_seconds")

        from ml.labeling import Document, FieldAnnotation, assign_labels

        doc = Document.from_dict(record.doc_json)
        doc.annotations = []
        truth: dict[str, str] = {}

        for corr in ser.validated_data["corrections"]:
            fname = corr["field"]
            pred = predicted.get(fname, {})
            value = (str(pred.get("raw", "")) if corr["accepted_as_is"]
                     else corr["corrected_value"])
            bbox = corr.get("bbox") or pred.get("bbox")
            Correction.objects.create(
                extraction=extraction,
                field=fname,
                predicted_value=str(pred.get("raw", "")),
                corrected_value=value,
                predicted_confidence=pred.get("confidence"),
                bbox=corr.get("bbox"),
                accepted_as_is=corr["accepted_as_is"],
                review_seconds=review_seconds,
            )
            if value and bbox:
                truth[fname] = value
                doc.annotations.append(FieldAnnotation(
                    field=fname, value=value, page=bbox.get("page", 0),
                    x0=bbox["x0"], y0=bbox["y0"], x1=bbox["x1"], y1=bbox["y1"],
                ))

        assign_labels(doc)
        doc.meta["truth"] = truth
        record.doc_json = doc.to_dict()
        record.status = InvoiceDocument.Status.VERIFIED
        record.verified_at = timezone.now()
        record.save()
        return Response({"status": record.status,
                         "gold_annotations": len(doc.annotations)})
