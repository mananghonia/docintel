from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from training.models import ModelVersion, TrainingRun
from training.tasks import retrain


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = ["id", "version", "metrics", "n_training_docs",
                  "is_champion", "created_at"]


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = ["id", "triggered_by", "outcome", "n_train_docs",
                  "n_holdout_docs", "champion_f1", "challenger_f1",
                  "detail", "created_at"]


class ModelVersionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer


class TrainingRunViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = TrainingRun.objects.all()
    serializer_class = TrainingRunSerializer

    @action(detail=False, methods=["post"])
    def trigger(self, request):
        result = retrain.delay(triggered_by="manual")
        return Response({"task_id": str(result)})
