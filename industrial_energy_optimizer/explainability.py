"""SHAP and LIME explainability for machine anomaly classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import lime.lime_tabular

    HAS_LIME = True
except Exception:
    HAS_LIME = False


@dataclass
class ExplainSample:
    row_index: int
    shap_top_features: List[Tuple[str, float]]
    lime_weights: List[Tuple[str, float]]
    predicted_proba: float


def explain_anomalies(
    clf: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    row_positions: np.ndarray,
    max_samples: int = 5,
    proba_threshold: float = 0.35,
    random_state: int = 42,
) -> Tuple[List[ExplainSample], str]:
    """
    Explain high-risk rows: SHAP (Tree) + LIME tabular.
    `row_positions` are indices into the original dataframe for display only.
    """
    proba = clf.predict_proba(X_explain)[:, 1]
    order = np.argsort(-proba)
    picked: List[ExplainSample] = []
    notes: List[str] = []

    if not HAS_SHAP:
        notes.append("SHAP not available in this environment.")
    if not HAS_LIME:
        notes.append("LIME not available in this environment.")

    bg = (
        shap.sample(X_train, min(200, len(X_train)), random_state=random_state)
        if HAS_SHAP
        else None
    )
    explainer = shap.TreeExplainer(clf, bg) if HAS_SHAP and bg is not None else None

    lime_exp = None
    if HAS_LIME:
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=list(X_train.columns),
            class_names=["normal", "abnormal"],
            mode="classification",
            random_state=random_state,
        )

    n_take = 0
    for j in order:
        if n_take >= max_samples:
            break
        if proba[j] < proba_threshold and n_take > 0:
            break
        row_vec = X_explain.iloc[[j]].values
        pred_p = float(proba[j])

        shap_pairs: List[Tuple[str, float]] = []
        if explainer is not None:
            sv = explainer.shap_values(row_vec)
            if isinstance(sv, list):
                arr = sv[1][0]
            else:
                arr = sv[0]
            names = list(X_explain.columns)
            for name, val in sorted(zip(names, arr), key=lambda x: -abs(x[1]))[:5]:
                shap_pairs.append((name, float(val)))

        lime_pairs: List[Tuple[str, float]] = []
        if lime_exp is not None:
            ex = lime_exp.explain_instance(
                row_vec[0],
                clf.predict_proba,
                num_features=min(6, X_explain.shape[1]),
            )
            lime_pairs = [(str(a), float(b)) for a, b in ex.as_list()]

        ri = int(row_positions[j]) if j < len(row_positions) else int(j)
        picked.append(
            ExplainSample(
                row_index=ri,
                shap_top_features=shap_pairs,
                lime_weights=lime_pairs,
                predicted_proba=pred_p,
            )
        )
        n_take += 1

    if not picked and len(order):
        j = int(order[0])
        row_vec = X_explain.iloc[[j]].values
        shap_pairs, lime_pairs = [], []
        if explainer is not None:
            sv = explainer.shap_values(row_vec)
            if isinstance(sv, list):
                arr = sv[1][0]
            else:
                arr = sv[0]
            names = list(X_explain.columns)
            for name, val in sorted(zip(names, arr), key=lambda x: -abs(x[1]))[:5]:
                shap_pairs.append((name, float(val)))
        if lime_exp is not None:
            ex = lime_exp.explain_instance(
                row_vec[0],
                clf.predict_proba,
                num_features=min(6, X_explain.shape[1]),
            )
            lime_pairs = [(str(a), float(b)) for a, b in ex.as_list()]
        picked.append(
            ExplainSample(
                row_index=int(row_positions[j]) if j < len(row_positions) else j,
                shap_top_features=shap_pairs,
                lime_weights=lime_pairs,
                predicted_proba=float(proba[j]),
            )
        )

    return picked, " | ".join(notes) if notes else "OK"
