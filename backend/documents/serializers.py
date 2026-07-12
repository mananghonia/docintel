from rest_framework import serializers

from documents.models import Correction, Extraction, InvoiceDocument


class ExtractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Extraction
        fields = ["id", "fields", "avg_confidence", "min_confidence",
                  "needs_review", "latency_ms", "created_at"]


class DocumentListSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvoiceDocument
        fields = ["id", "source", "status", "is_holdout", "created_at", "error"]


class DocumentDetailSerializer(serializers.ModelSerializer):
    latest_extraction = serializers.SerializerMethodField()

    class Meta:
        model = InvoiceDocument
        fields = ["id", "source", "status", "is_holdout", "created_at",
                  "error", "doc_json", "latest_extraction"]

    def get_latest_extraction(self, obj):
        ext = obj.extractions.first()  # ordering: newest first
        return ExtractionSerializer(ext).data if ext else None


class CorrectionInSerializer(serializers.Serializer):
    field = serializers.CharField()
    corrected_value = serializers.CharField(allow_blank=True)
    bbox = serializers.DictField(required=False, allow_null=True)
    accepted_as_is = serializers.BooleanField(default=False)


class ReviewSerializer(serializers.Serializer):
    corrections = CorrectionInSerializer(many=True)
    review_seconds = serializers.FloatField(required=False, allow_null=True)


class CorrectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Correction
        fields = ["id", "field", "predicted_value", "corrected_value",
                  "predicted_confidence", "bbox", "accepted_as_is",
                  "review_seconds", "created_at"]
