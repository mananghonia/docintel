from django.db import models


class ModelVersion(models.Model):
    version = models.PositiveIntegerField(unique=True)
    artifact_path = models.CharField(max_length=500)
    # {"field_macro_f1": ..., "token_macro_f1": ..., "ece": ..., "temperature": ...}
    metrics = models.JSONField(default=dict)
    n_training_docs = models.PositiveIntegerField(default=0)
    is_champion = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version"]

    def __str__(self):
        star = "*" if self.is_champion else " "
        return f"v{self.version}{star} f1={self.metrics.get('field_macro_f1', 0):.3f}"


class TrainingRun(models.Model):
    class Outcome(models.TextChoices):
        RUNNING = "running"
        PROMOTED = "promoted"
        REJECTED = "rejected"     # challenger lost on the frozen holdout — logged, not hidden
        FAILED = "failed"
        SKIPPED = "skipped"       # not enough new verified docs

    triggered_by = models.CharField(max_length=40, default="beat")  # beat|manual
    outcome = models.CharField(max_length=20, choices=Outcome.choices,
                               default=Outcome.RUNNING)
    n_train_docs = models.PositiveIntegerField(default=0)
    n_holdout_docs = models.PositiveIntegerField(default=0)
    champion_f1 = models.FloatField(null=True, blank=True)
    challenger_f1 = models.FloatField(null=True, blank=True)
    model_version = models.ForeignKey(ModelVersion, null=True, blank=True,
                                      on_delete=models.SET_NULL)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
