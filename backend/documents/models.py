import uuid

from django.db import models


class InvoiceDocument(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded"
        PROCESSING = "processing"
        EXTRACTED = "extracted"       # auto-accepted, no review needed
        NEEDS_REVIEW = "needs_review"
        VERIFIED = "verified"         # human signed off -> training candidate
        FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to="uploads/%Y/%m/", null=True, blank=True)
    source = models.CharField(max_length=20, default="upload")  # upload|synthetic
    status = models.CharField(max_length=20, choices=Status.choices,
                              default=Status.UPLOADED)
    # Tokens + geometry + (once verified) gold annotations, as ml.labeling.Document dict.
    doc_json = models.JSONField(null=True, blank=True)
    # Frozen holdout docs NEVER enter training; the champion/challenger gate
    # is scored on them.
    is_holdout = models.BooleanField(default=False)
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    verified_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class Extraction(models.Model):
    """One model pass over one document: per-field values + confidences."""

    document = models.ForeignKey(InvoiceDocument, on_delete=models.CASCADE,
                                 related_name="extractions")
    model_version = models.ForeignKey("training.ModelVersion", null=True,
                                      blank=True, on_delete=models.SET_NULL)
    # [{field, raw, value, confidence, flags, bbox: {x0,y0,x1,y1,page}}]
    fields = models.JSONField(default=list)
    token_tags = models.JSONField(default=list)  # predicted tag per token
    avg_confidence = models.FloatField(default=0.0)
    min_confidence = models.FloatField(default=0.0)
    needs_review = models.BooleanField(default=True)
    latency_ms = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        get_latest_by = "created_at"


class Correction(models.Model):
    """A human's delta on one field — the training signal.

    predicted_value empty + a drawn bbox = the model missed the field
    entirely: a hard negative, the most valuable example there is.
    """

    extraction = models.ForeignKey(Extraction, on_delete=models.CASCADE,
                                   related_name="corrections")
    field = models.CharField(max_length=40)
    predicted_value = models.TextField(blank=True, default="")
    corrected_value = models.TextField()
    predicted_confidence = models.FloatField(null=True, blank=True)
    # Reviewer-drawn box for missed fields: {x0,y0,x1,y1,page}
    bbox = models.JSONField(null=True, blank=True)
    accepted_as_is = models.BooleanField(default=False)
    review_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
