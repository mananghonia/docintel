"""Tier 1: classical models over engineered features.

Six models, one comparison table (via ml.evaluate.compare_models). The point
is not that XGBoost wins — it will — but being able to say WHY each of the
others loses: NB's independence assumption is false for correlated geometry
features, KNN collapses in 120-D, linear LogReg can't represent "amount-shaped
AND right-of-'Total'" interactions, trees can.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone


class LabelEncoded(BaseEstimator, ClassifierMixin):
    """Remap labels to 0..k-1 for estimators (XGBoost) that require it.

    Tag ids are sparse in practice — some I- tags never occur — so the raw
    y is non-contiguous. predict() returns original tag ids; predict_proba
    columns follow self.classes_ (original ids, sorted).
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder

        self.le_ = LabelEncoder().fit(y)
        self.model_ = clone(self.estimator).fit(X, self.le_.transform(y))
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.model_.predict(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


def make_models(fast: bool = False) -> dict:
    """Return name -> unfitted sklearn-style estimator.

    fast=True shrinks the expensive ones for smoke tests.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from xgboost import XGBClassifier

    n_estimators = 100 if fast else 300

    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        ),
        "naive_bayes": GaussianNB(),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight="balanced", dual="auto"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=0,
        ),
        "xgboost": LabelEncoded(XGBClassifier(
            n_estimators=n_estimators, max_depth=7, learning_rate=0.15,
            subsample=0.9, colsample_bytree=0.8, tree_method="hist",
            n_jobs=-1, random_state=0, verbosity=0,
        )),
    }


def feature_importance(model, top_k: int = 25) -> list[tuple[str, float]]:
    """Top-k (feature_name, importance) for tree models."""
    from ml.features import FEATURE_NAMES

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise ValueError("Model has no feature_importances_ (tree models only)")
    pairs = sorted(zip(FEATURE_NAMES, imp), key=lambda p: -p[1])
    return [(n, float(v)) for n, v in pairs[:top_k]]
