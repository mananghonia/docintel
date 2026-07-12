"""The self-improving loop: retrain on human-verified documents, promote only
if the challenger beats the champion on a frozen holdout it has never seen."""

from celery import shared_task
from django.conf import settings


def _load_docs(queryset):
    from ml.labeling import Document

    return [Document.from_dict(r.doc_json) for r in queryset if r.doc_json]


@shared_task
def maybe_retrain() -> str:
    """Beat entrypoint: retrain when enough new verified docs accumulated."""
    from documents.models import InvoiceDocument
    from training.models import ModelVersion, TrainingRun

    champion = ModelVersion.objects.filter(is_champion=True).first()
    since = champion.created_at if champion else None
    qs = InvoiceDocument.objects.filter(
        status=InvoiceDocument.Status.VERIFIED, is_holdout=False)
    new_docs = qs.filter(verified_at__gt=since).count() if since else qs.count()

    if new_docs < settings.RETRAIN_MIN_NEW_DOCS:
        TrainingRun.objects.create(
            outcome=TrainingRun.Outcome.SKIPPED,
            detail=f"{new_docs}/{settings.RETRAIN_MIN_NEW_DOCS} new verified docs")
        return f"skipped ({new_docs} new docs)"
    return retrain(triggered_by="beat")


@shared_task
def retrain(triggered_by: str = "manual") -> str:
    import joblib

    from documents.models import InvoiceDocument
    from ml.calibration import TemperatureScaler, probs_to_logits
    from ml.evaluate import evaluate_field_extraction
    from ml.features import featurize_dataset, featurize_document
    from ml.labeling import ID2TAG
    from ml.models_classical import make_models
    from training.models import ModelVersion, TrainingRun

    run = TrainingRun.objects.create(triggered_by=triggered_by)
    try:
        train_docs = _load_docs(InvoiceDocument.objects.filter(
            status=InvoiceDocument.Status.VERIFIED, is_holdout=False))
        holdout_docs = _load_docs(InvoiceDocument.objects.filter(
            status=InvoiceDocument.Status.VERIFIED, is_holdout=True))
        run.n_train_docs, run.n_holdout_docs = len(train_docs), len(holdout_docs)

        if len(train_docs) < 10 or len(holdout_docs) < 3:
            run.outcome = TrainingRun.Outcome.SKIPPED
            run.detail = "not enough verified docs (need 10 train / 3 holdout)"
            run.save()
            return run.detail

        # Train challenger: XGBoost + temperature calibration on a val split.
        n_val = max(len(train_docs) // 5, 2)
        fit_docs, val_docs = train_docs[n_val:], train_docs[:n_val]
        X, y, _ = featurize_dataset(fit_docs)
        model = make_models()["xgboost"]
        model.fit(X, y)

        Xv, yv, _ = featurize_dataset(val_docs)
        class_pos = {int(c): i for i, c in enumerate(model.classes_)}
        import numpy as np
        yv_enc = np.array([class_pos.get(int(t), 0) for t in yv])
        scaler = TemperatureScaler().fit(probs_to_logits(model.predict_proba(Xv)), yv_enc)

        # Score challenger and current champion on the FROZEN holdout.
        def holdout_f1(m) -> float:
            tag_lists = [[ID2TAG[int(p)] for p in m.predict(featurize_document(d))]
                         for d in holdout_docs]
            return evaluate_field_extraction(holdout_docs, tag_lists).macro_f1()

        challenger_f1 = holdout_f1(model)
        run.challenger_f1 = challenger_f1

        champion_row = ModelVersion.objects.filter(is_champion=True).first()
        champion_f1 = None
        if champion_row:
            champ = joblib.load(champion_row.artifact_path)
            champion_f1 = holdout_f1(champ["model"])
        run.champion_f1 = champion_f1

        if champion_f1 is not None and \
                challenger_f1 <= champion_f1 + settings.CHALLENGER_MIN_IMPROVEMENT:
            run.outcome = TrainingRun.Outcome.REJECTED
            run.detail = (f"challenger {challenger_f1:.4f} did not beat "
                          f"champion {champion_f1:.4f} on frozen holdout")
            run.save()
            return run.detail

        # Promote.
        version = (champion_row.version + 1) if champion_row else 1
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        artifact = settings.MODEL_DIR / "champion.joblib"
        joblib.dump({"model": model, "scaler": scaler, "version": version}, artifact)

        ModelVersion.objects.filter(is_champion=True).update(is_champion=False)
        mv = ModelVersion.objects.create(
            version=version, artifact_path=str(artifact), is_champion=True,
            n_training_docs=len(train_docs),
            metrics={"field_macro_f1": challenger_f1,
                     "temperature": scaler.temperature},
        )
        run.model_version = mv
        run.outcome = TrainingRun.Outcome.PROMOTED
        run.detail = (f"v{version} promoted: {challenger_f1:.4f} vs "
                      f"{'none' if champion_f1 is None else f'{champion_f1:.4f}'}")
        run.save()
        _log_mlflow(mv, run)
        return run.detail
    except Exception as e:  # noqa: BLE001
        run.outcome = TrainingRun.Outcome.FAILED
        run.detail = str(e)
        run.save()
        raise


def _log_mlflow(model_version, run) -> None:
    """Best-effort MLflow logging; absence of a tracking server is not an error."""
    try:
        import mlflow

        with mlflow.start_run(run_name=f"v{model_version.version}"):
            mlflow.log_metrics({
                "field_macro_f1": model_version.metrics["field_macro_f1"],
                "n_train_docs": run.n_train_docs,
            })
    except Exception:  # noqa: BLE001
        pass
