"""The self-improving loop: retrain on human-verified documents, promote only
if the challenger beats the champion on a frozen holdout it has never seen."""

import os

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

    import numpy as np

    from documents.models import InvoiceDocument
    from ml.calibration import (TemperatureScaler, expected_calibration_error,
                                probs_to_logits)
    from ml.evaluate import (bootstrap_win_rate, evaluate_field_extraction,
                             per_document_field_f1)
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

        if len(train_docs) < settings.RETRAIN_MIN_TRAIN_DOCS or \
                len(holdout_docs) < settings.RETRAIN_MIN_HOLDOUT_DOCS:
            run.outcome = TrainingRun.Outcome.SKIPPED
            run.detail = (f"not enough verified docs (have {len(train_docs)}"
                          f"/{len(holdout_docs)}, need "
                          f"{settings.RETRAIN_MIN_TRAIN_DOCS}/"
                          f"{settings.RETRAIN_MIN_HOLDOUT_DOCS})")
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
        yv_enc = np.array([class_pos.get(int(t), 0) for t in yv])
        val_probs = model.predict_proba(Xv)
        scaler = TemperatureScaler().fit(probs_to_logits(val_probs), yv_enc)

        # #4: measure calibration honestly — ECE before/after on validation.
        correct = (val_probs.argmax(axis=1) == yv_enc).astype(float)
        ece_before = expected_calibration_error(val_probs.max(axis=1), correct)
        ece_after = expected_calibration_error(
            scaler.transform(probs_to_logits(val_probs)).max(axis=1), correct)

        # Per-document tags on the FROZEN holdout, for a paired bootstrap.
        def holdout_tags(m):
            return [[ID2TAG[int(p)] for p in m.predict(featurize_document(d))]
                    for d in holdout_docs]

        chal_tags = holdout_tags(model)
        challenger_f1 = evaluate_field_extraction(holdout_docs, chal_tags).macro_f1()
        run.challenger_f1 = challenger_f1

        champion_row = ModelVersion.objects.filter(is_champion=True).first()
        champion_f1, win_rate = None, 1.0
        # A champion row whose artifact is gone must not block promotion or
        # crash the run — treat it as "no champion" and let the challenger in.
        champ_bundle = None
        if champion_row and os.path.exists(champion_row.artifact_path):
            champ_bundle = joblib.load(champion_row.artifact_path)
        if champ_bundle is not None:
            champ_tags = holdout_tags(champ_bundle["model"])
            champion_f1 = evaluate_field_extraction(holdout_docs, champ_tags).macro_f1()
            # #1: don't promote on holdout noise. Require both a margin AND a
            # paired bootstrap that says the win is unlikely to be chance.
            win_rate = bootstrap_win_rate(
                per_document_field_f1(holdout_docs, chal_tags),
                per_document_field_f1(holdout_docs, champ_tags))
        run.champion_f1 = champion_f1

        if champion_f1 is not None and (
                challenger_f1 <= champion_f1 + settings.CHALLENGER_MIN_IMPROVEMENT
                or win_rate < settings.CHALLENGER_MIN_WIN_RATE):
            run.outcome = TrainingRun.Outcome.REJECTED
            run.detail = (f"challenger {challenger_f1:.4f} vs champion "
                          f"{champion_f1:.4f} (bootstrap win-rate {win_rate:.2f}) "
                          f"— not a significant improvement on frozen holdout")
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
                     "temperature": scaler.temperature,
                     "ece_before": ece_before, "ece_after": ece_after,
                     "holdout_win_rate": win_rate},
        )
        run.model_version = mv
        run.outcome = TrainingRun.Outcome.PROMOTED
        run.detail = (f"v{version} promoted: {challenger_f1:.4f} vs "
                      f"{'none' if champion_f1 is None else f'{champion_f1:.4f}'}"
                      f" (win-rate {win_rate:.2f}, ECE {ece_before:.3f}->{ece_after:.3f})")
        run.save()
        _log_mlflow(mv, run)
        return run.detail
    except Exception as e:  # noqa: BLE001
        run.outcome = TrainingRun.Outcome.FAILED
        run.detail = str(e)
        run.save()
        raise


def _log_mlflow(model_version, run) -> None:
    """Log the promoted model to MLflow: params, metrics, and the artifact.

    Best-effort — if MLFLOW_TRACKING_URI is unset or the server is unreachable,
    log locally to ./mlruns; any failure is swallowed so training never depends
    on the tracking server being up.
    """
    try:
        import os

        import mlflow

        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("docintel-champion")
        m = model_version.metrics
        with mlflow.start_run(run_name=f"v{model_version.version}"):
            mlflow.log_params({
                "version": model_version.version,
                "n_train_docs": run.n_train_docs,
                "n_holdout_docs": run.n_holdout_docs,
                "triggered_by": run.triggered_by,
                "temperature": m.get("temperature"),
            })
            mlflow.log_metrics({
                "field_macro_f1": m.get("field_macro_f1", 0.0),
                "ece_before": m.get("ece_before", 0.0),
                "ece_after": m.get("ece_after", 0.0),
                "holdout_win_rate": m.get("holdout_win_rate", 0.0),
                "champion_f1": run.champion_f1 or 0.0,
                "challenger_f1": run.challenger_f1 or 0.0,
            })
            if os.path.exists(model_version.artifact_path):
                mlflow.log_artifact(model_version.artifact_path, "model")
    except Exception:  # noqa: BLE001
        pass
