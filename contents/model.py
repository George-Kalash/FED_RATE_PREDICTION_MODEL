"""Stage 4: train a leakage-aware hierarchical FOMC decision model.

The model first estimates hold versus change.  Conditional on a change, a
second logistic regression estimates cut versus hike.  A decision policy whose
thresholds are selected only from chronological training-fold predictions then
combines the two probabilities and records every override in predictions.csv.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CHANGE_THRESHOLD_VALUES,
    CUT_OVERRIDE_DIRECTION_VALUES,
    CUT_OVERRIDE_JOINT_VALUES,
    CUT_OVERRIDE_MIN_CHANGE_VALUES,
    CV_SPLITS,
    DECISION_CLASSES,
    DECISION_POLICY_SCORING,
    DIAGNOSTIC_MODEL_SCORING,
    DIAGNOSTIC_TARGET,
    DIRECTION_CUT_THRESHOLD_VALUES,
    DIRECTION_CUT_WEIGHT_OPTIONS,
    DIRECTION_TARGET,
    FEATURE_COLUMNS,
    FEATURE_PANEL_PATH,
    LOGISTIC_C_VALUES,
    LOGISTIC_MAX_ITERATIONS,
    MODEL_COEFFICIENTS_PATH,
    MODEL_METRICS_PATH,
    MODEL_PREDICTIONS_PATH,
    MODEL_SELECTION_SCORE_TOLERANCE,
    MODEL_TEST_FRACTION,
    PRIMARY_CLASS_WEIGHT_OPTIONS,
    PRIMARY_MODEL_SCORING,
    PRIMARY_TARGET,
    RANDOM_STATE,
)
from features import validate_feature_panel


def _json_ready(value: Any) -> Any:
    """Recursively convert numpy values and mapping keys for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _observed_class_balanced_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Calculate mean recall without warning on one-class validation windows."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    observed = np.unique(truth)
    return float(
        np.mean(
            [
                np.mean(predicted[truth == label] == label)
                for label in observed
            ]
        )
    )


def _pipeline(*, class_weight: Any = None, C: float = 1.0) -> Pipeline:
    """Create the scaler/model pipeline used in every fit."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=LOGISTIC_MAX_ITERATIONS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _class_complete_cv(y: pd.Series) -> TimeSeriesSplit:
    """Return forward folds whose training prefix contains all target classes."""
    if len(y) < CV_SPLITS + 2 or y.nunique() < 2:
        raise ValueError("Target is too short or contains fewer than two classes")
    required = set(y.tolist())
    first_complete: int | None = None
    for size in range(2, len(y) + 1):
        if set(y.iloc[:size]) == required:
            first_complete = size
            break
    if first_complete is None:
        raise ValueError("No training prefix contains all target classes")
    maximum_test_size = (len(y) - first_complete) // CV_SPLITS
    if maximum_test_size < 1:
        raise ValueError("Not enough rows for class-complete chronological CV")
    test_size = min(len(y) // (CV_SPLITS + 1), maximum_test_size)
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=test_size)
    for train_index, _ in splitter.split(y):
        if set(y.iloc[train_index]) != required:
            raise ValueError("A chronological CV training fold lacks a class")
    return splitter


def load_feature_panel() -> pd.DataFrame:
    """Load the configured feature panel."""
    if not FEATURE_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {FEATURE_PANEL_PATH}; run features.py first"
        )
    panel = pd.read_csv(FEATURE_PANEL_PATH)
    panel["meeting_date"] = pd.to_datetime(panel["meeting_date"], errors="coerce")
    validate_feature_panel(panel)
    return panel


def chronological_train_test_split(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the newest configured fraction as a never-tuned holdout."""
    validate_feature_panel(panel)
    if not 0 < MODEL_TEST_FRACTION < 1:
        raise ValueError("MODEL_TEST_FRACTION must be between zero and one")
    test_size = math.ceil(len(panel) * MODEL_TEST_FRACTION)
    split = len(panel) - test_size
    train = panel.iloc[:split].copy().reset_index(drop=True)
    test = panel.iloc[split:].copy().reset_index(drop=True)
    if train.empty or test.empty or train.meeting_date.max() >= test.meeting_date.min():
        raise ValueError("Chronological train/test split is invalid")
    for name, part in (("train", train), ("test", test)):
        if set(part[PRIMARY_TARGET]) != {0, 1}:
            raise ValueError(f"{name} lacks a binary target class")
        if set(part[DIAGNOSTIC_TARGET]) != set(DECISION_CLASSES):
            raise ValueError(f"{name} lacks a decision class")
    return train, test


def _tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    class_weights: Iterable[Any],
    scoring: str,
) -> GridSearchCV:
    """Tune regularization and class weights with chronological CV."""
    if X.empty or len(X) != len(y) or X.isna().any().any():
        raise ValueError("Training features/target are empty, unequal, or missing")
    scorer: Any = scoring
    if scoring == "balanced_accuracy":
        scorer = make_scorer(_observed_class_balanced_accuracy)
    search = GridSearchCV(
        _pipeline(),
        {
            "logistic_regression__C": list(LOGISTIC_C_VALUES),
            "logistic_regression__class_weight": list(class_weights),
        },
        scoring=scorer,
        cv=_class_complete_cv(y.reset_index(drop=True)),
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def _positive_probability(estimator: Any, X: pd.DataFrame, label: Any) -> np.ndarray:
    """Return the probability column for a named class."""
    pipeline = getattr(estimator, "best_estimator_", estimator)
    classes = list(pipeline.named_steps["logistic_regression"].classes_)
    if label not in classes:
        raise ValueError(f"Estimator does not contain class {label!r}: {classes}")
    return pipeline.predict_proba(X)[:, classes.index(label)]


def _fixed_pipeline(best_parameters: dict[str, Any]) -> Pipeline:
    """Recreate a tuned pipeline for an out-of-fold fit."""
    return _pipeline(
        C=float(best_parameters["logistic_regression__C"]),
        class_weight=best_parameters["logistic_regression__class_weight"],
    )


def _make_oof_probabilities(
    train: pd.DataFrame,
    primary_parameters: dict[str, Any],
    direction_parameters: dict[str, Any],
) -> pd.DataFrame:
    """Generate forward-only probabilities used to select policy thresholds."""
    X = train.loc[:, FEATURE_COLUMNS]
    y_change = train[PRIMARY_TARGET].astype(int)
    probabilities = pd.DataFrame(
        {
            "actual_decision": train[DIAGNOSTIC_TARGET].astype(str),
            "probability_change": np.nan,
            "probability_cut_given_change": np.nan,
        },
        index=train.index,
    )
    for fit_index, validation_index in _class_complete_cv(y_change).split(X):
        primary = _fixed_pipeline(primary_parameters)
        primary.fit(X.iloc[fit_index], y_change.iloc[fit_index])
        probabilities.loc[validation_index, "probability_change"] = (
            _positive_probability(primary, X.iloc[validation_index], 1)
        )

        direction_train = train.iloc[fit_index]
        direction_train = direction_train.loc[direction_train[PRIMARY_TARGET].eq(1)]
        direction_y = direction_train[DIAGNOSTIC_TARGET].astype(str)
        if set(direction_y) != {"cut", "hike"}:
            raise ValueError("An OOF direction-training fold lacks cut or hike")
        direction = _fixed_pipeline(direction_parameters)
        direction.fit(direction_train.loc[:, FEATURE_COLUMNS], direction_y)
        probabilities.loc[validation_index, "probability_cut_given_change"] = (
            _positive_probability(direction, X.iloc[validation_index], "cut")
        )
    complete = probabilities.dropna().copy()
    if complete.empty:
        raise ValueError("Chronological CV produced no policy-training predictions")
    return complete.reset_index(drop=True)


def apply_decision_policy(
    probability_change: np.ndarray,
    probability_cut_given_change: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Combine model probabilities and apply the auditable obvious-cut rule."""
    p_change = np.asarray(probability_change, dtype=float)
    p_cut_direction = np.asarray(probability_cut_given_change, dtype=float)
    if p_change.shape != p_cut_direction.shape or p_change.ndim != 1:
        raise ValueError("Policy probability arrays must be equal one-dimensional arrays")
    if (
        np.isnan(p_change).any()
        or np.isnan(p_cut_direction).any()
        or np.any((p_change < 0) | (p_change > 1))
        or np.any((p_cut_direction < 0) | (p_cut_direction > 1))
    ):
        raise ValueError("Policy probabilities must be finite values from zero to one")

    required = {
        "change_threshold",
        "direction_cut_threshold",
        "override_min_change_probability",
        "override_min_direction_cut_probability",
        "override_min_joint_cut_probability",
    }
    if set(thresholds) != required:
        raise ValueError(f"Policy thresholds must be exactly {sorted(required)}")

    predicted_change = p_change >= thresholds["change_threshold"]
    predicted_direction = np.where(
        p_cut_direction >= thresholds["direction_cut_threshold"], "cut", "hike"
    )
    raw_decision = np.where(predicted_change, predicted_direction, "hold")
    p_cut = p_change * p_cut_direction
    p_hike = p_change * (1.0 - p_cut_direction)
    p_hold = 1.0 - p_change

    obvious_cut = (
        (p_change >= thresholds["override_min_change_probability"])
        & (
            p_cut_direction
            >= thresholds["override_min_direction_cut_probability"]
        )
        & (p_cut >= thresholds["override_min_joint_cut_probability"])
    )
    final_decision = np.where(obvious_cut, "cut", raw_decision)
    override = obvious_cut & (raw_decision != "cut")
    reason = np.where(
        override,
        "cut probabilities exceeded all training-selected override gates",
        "",
    )
    return pd.DataFrame(
        {
            "raw_predicted_is_change": predicted_change.astype("int8"),
            "raw_direction": predicted_direction,
            "raw_model_decision": raw_decision,
            "final_decision": final_decision,
            "probability_change": p_change,
            "probability_cut_given_change": p_cut_direction,
            "probability_hike_given_change": 1.0 - p_cut_direction,
            "probability_cut": p_cut,
            "probability_hold": p_hold,
            "probability_hike": p_hike,
            "obvious_cut_signal": obvious_cut,
            "cut_override_triggered": override,
            "override_reason": reason,
        }
    )


def _policy_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Score the decision policy using the configured training objective."""
    if DECISION_POLICY_SCORING != "f1_macro":
        raise ValueError(
            "Only f1_macro is currently implemented for DECISION_POLICY_SCORING"
        )
    return float(f1_score(y_true, y_pred, labels=DECISION_CLASSES, average="macro"))


def select_decision_policy(oof: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    """Select all decision and override thresholds from training OOF rows."""
    required = {
        "actual_decision",
        "probability_change",
        "probability_cut_given_change",
    }
    if not required.issubset(oof.columns) or oof.empty:
        raise ValueError("OOF policy data is missing required columns")
    y = oof["actual_decision"].astype(str)
    if set(y) != set(DECISION_CLASSES):
        raise ValueError("OOF policy rows do not contain every decision class")

    best_thresholds: dict[str, float] | None = None
    best_policy: pd.DataFrame | None = None
    best_rank: tuple[float, float, float, float, int] | None = None
    for values in product(
        CHANGE_THRESHOLD_VALUES,
        DIRECTION_CUT_THRESHOLD_VALUES,
        CUT_OVERRIDE_MIN_CHANGE_VALUES,
        CUT_OVERRIDE_DIRECTION_VALUES,
        CUT_OVERRIDE_JOINT_VALUES,
    ):
        thresholds = {
            "change_threshold": float(values[0]),
            "direction_cut_threshold": float(values[1]),
            "override_min_change_probability": float(values[2]),
            "override_min_direction_cut_probability": float(values[3]),
            "override_min_joint_cut_probability": float(values[4]),
        }
        policy = apply_decision_policy(
            oof["probability_change"].to_numpy(),
            oof["probability_cut_given_change"].to_numpy(),
            thresholds,
        )
        predicted = policy["final_decision"]
        cut_mask = y.eq("cut")
        cut_recall = float(predicted[cut_mask].eq("cut").mean())
        rank = (
            _policy_score(y, predicted),
            cut_recall,
            float(balanced_accuracy_score(y, predicted)),
            float(accuracy_score(y, predicted)),
            -int(policy["cut_override_triggered"].sum()),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thresholds = thresholds
            best_policy = policy
    if best_thresholds is None or best_policy is None or best_rank is None:
        raise RuntimeError("Decision-policy search evaluated no candidates")
    audit = {
        "selection_source": "chronological training out-of-fold predictions",
        "sample_count": int(len(oof)),
        "scoring": DECISION_POLICY_SCORING,
        "macro_f1": best_rank[0],
        "cut_recall": best_rank[1],
        "balanced_accuracy": best_rank[2],
        "accuracy": best_rank[3],
        "override_count": int(best_policy["cut_override_triggered"].sum()),
        "candidates_evaluated": int(
            len(CHANGE_THRESHOLD_VALUES)
            * len(DIRECTION_CUT_THRESHOLD_VALUES)
            * len(CUT_OVERRIDE_MIN_CHANGE_VALUES)
            * len(CUT_OVERRIDE_DIRECTION_VALUES)
            * len(CUT_OVERRIDE_JOINT_VALUES)
        ),
    }
    return best_thresholds, audit


def _select_primary_model_and_policy(
    train: pd.DataFrame,
    primary_search: GridSearchCV,
    direction_parameters: dict[str, Any],
) -> tuple[GridSearchCV, dict[str, float], dict[str, Any]]:
    """Jointly choose the change model and policy on forward training predictions."""
    candidates: list[
        tuple[dict[str, Any], dict[str, float], dict[str, Any]]
    ] = []
    for C, class_weight in product(
        LOGISTIC_C_VALUES, PRIMARY_CLASS_WEIGHT_OPTIONS
    ):
        parameters = {
            "logistic_regression__C": float(C),
            "logistic_regression__class_weight": class_weight,
        }
        oof = _make_oof_probabilities(train, parameters, direction_parameters)
        thresholds, audit = select_decision_policy(oof)
        candidates.append((parameters, thresholds, audit))
    if not candidates:
        raise RuntimeError("Joint primary-model/policy search produced no result")

    maximum_score = max(float(item[2]["macro_f1"]) for item in candidates)
    eligible = [
        item
        for item in candidates
        if float(item[2]["macro_f1"])
        >= maximum_score - MODEL_SELECTION_SCORE_TOLERANCE
    ]
    best_parameters, best_thresholds, best_audit = max(
        eligible,
        key=lambda item: (
            -float(item[0]["logistic_regression__C"]),
            float(item[2]["macro_f1"]),
            float(item[2]["cut_recall"]),
            float(item[2]["balanced_accuracy"]),
            float(item[2]["accuracy"]),
        ),
    )

    selected = _fixed_pipeline(best_parameters)
    selected.fit(train.loc[:, FEATURE_COLUMNS], train[PRIMARY_TARGET].astype(int))
    primary_search.best_estimator_ = selected
    primary_search.best_params_ = best_parameters
    primary_search.best_score_ = float(best_audit["macro_f1"])
    best_audit["primary_parameter_candidates_evaluated"] = int(
        len(LOGISTIC_C_VALUES) * len(PRIMARY_CLASS_WEIGHT_OPTIONS)
    )
    best_audit["model_selection"] = (
        "Choose the strongest regularization within the configured tolerance "
        "of the best three-class OOF macro F1; then break ties by policy metrics"
    )
    best_audit["maximum_candidate_macro_f1"] = maximum_score
    best_audit["score_tolerance"] = MODEL_SELECTION_SCORE_TOLERANCE
    return primary_search, best_thresholds, best_audit


def _evaluate(
    y_true: pd.Series,
    y_pred: pd.Series,
    probabilities: np.ndarray,
    labels: list[Any],
) -> dict[str, Any]:
    """Build a common metrics block from explicit policy predictions."""
    truth = pd.Series(y_true).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)
    probability_array = np.asarray(probabilities, dtype=float)
    if len(truth) == 0 or len(truth) != len(predicted):
        raise ValueError("Evaluation targets and predictions must be equally sized")
    if probability_array.shape != (len(truth), len(labels)):
        raise ValueError("Evaluation probability matrix has the wrong shape")
    if not np.allclose(probability_array.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Evaluation probabilities do not sum to one")
    majority = truth.value_counts().idxmax()
    majority_prediction = pd.Series([majority] * len(truth))
    metrics: dict[str, Any] = {
        "sample_count": int(len(truth)),
        "class_counts": {
            str(label): int(truth.eq(label).sum()) for label in labels
        },
        "labels": _json_ready(labels),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "majority_baseline": {
            "class": _json_ready(majority),
            "accuracy": float(accuracy_score(truth, majority_prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(truth, majority_prediction)
            ),
        },
        "macro": {
            "precision": float(
                precision_score(
                    truth, predicted, labels=labels, average="macro", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    truth, predicted, labels=labels, average="macro", zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    truth, predicted, labels=labels, average="macro", zero_division=0
                )
            ),
        },
        "weighted": {
            "precision": float(
                precision_score(
                    truth,
                    predicted,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    truth,
                    predicted,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    truth,
                    predicted,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
        },
        "confusion_matrix": {
            "labels": _json_ready(labels),
            "values": confusion_matrix(truth, predicted, labels=labels).astype(int).tolist(),
        },
        "probability": {
            "log_loss": float(
                -np.mean(
                    np.log(
                        np.clip(
                            probability_array[
                                np.arange(len(truth)),
                                truth.map({label: index for index, label in enumerate(labels)}).to_numpy(),
                            ],
                            1e-15,
                            1.0,
                        )
                    )
                )
            )
        },
    }
    if len(labels) == 2 and set(truth) == set(labels):
        positive = labels[1]
        binary_truth = truth.eq(positive).astype(int)
        metrics["probability"].update(
            {
                "roc_auc": float(roc_auc_score(binary_truth, probability_array[:, 1])),
                "brier_score": float(
                    brier_score_loss(binary_truth, probability_array[:, 1])
                ),
                "positive_class": _json_ready(positive),
            }
        )
    elif len(labels) > 2 and set(truth) == set(labels):
        sorted_labels = sorted(labels)
        sorted_probabilities = probability_array[
            :, [labels.index(label) for label in sorted_labels]
        ]
        metrics["probability"]["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                truth,
                sorted_probabilities,
                labels=sorted_labels,
                multi_class="ovr",
                average="macro",
            )
        )
    return metrics


def _coefficients(
    estimator: GridSearchCV,
    *,
    target_name: str,
) -> pd.DataFrame:
    """Return standardized coefficients for both sides of a binary model."""
    pipeline = estimator.best_estimator_
    classifier = pipeline.named_steps["logistic_regression"]
    classes = list(classifier.classes_)
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercepts = np.asarray(classifier.intercept_, dtype=float)
    if len(classes) != 2 or coefficients.shape != (1, len(FEATURE_COLUMNS)):
        raise ValueError("Hierarchical component must be a fitted binary model")
    rows: list[dict[str, Any]] = []
    for class_index, class_label in enumerate(classes):
        sign = -1.0 if class_index == 0 else 1.0
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            coefficient = sign * float(coefficients[0, feature_index])
            rows.append(
                {
                    "target": target_name,
                    "model": "standard_scaler_logistic_regression",
                    "class": _json_ready(class_label),
                    "feature": feature,
                    "standardized_coefficient": coefficient,
                    "odds_ratio_per_1sd": float(np.exp(coefficient)),
                    "intercept": sign * float(intercepts[0]),
                    "regularization_C": float(classifier.C),
                    "class_weight": json.dumps(_json_ready(classifier.class_weight)),
                }
            )
    return pd.DataFrame(rows)


def _model_summary(
    search: GridSearchCV,
    y_train: pd.Series,
    *,
    target_name: str,
    scoring: str,
) -> dict[str, Any]:
    """Describe a tuned model before holdout metrics are attached."""
    return {
        "target": target_name,
        "scoring": scoring,
        "train_sample_count": int(len(y_train)),
        "train_class_counts": {
            str(label): int(count) for label, count in y_train.value_counts().items()
        },
        "cv_best_score": float(search.best_score_),
        "best_parameters": _json_ready(search.best_params_),
        "cv_splits": int(search.n_splits_),
    }


def _build_prediction_table(
    test: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Add dates and actual decisions to the complete policy audit table."""
    predictions = policy.copy()
    predictions.insert(
        0, "meeting_date", pd.to_datetime(test["meeting_date"]).reset_index(drop=True)
    )
    predictions.insert(1, "dataset_split", "chronological_holdout")
    predictions.insert(
        2, "actual_is_change", test[PRIMARY_TARGET].astype("int8").to_numpy()
    )
    predictions.insert(
        3,
        "predicted_is_change",
        predictions["final_decision"].ne("hold").astype("int8"),
    )
    predictions.insert(
        6,
        "actual_decision",
        test[DIAGNOSTIC_TARGET].astype(str).to_numpy(),
    )
    predictions.insert(7, "predicted_decision", predictions["final_decision"])
    if predictions["meeting_date"].isna().any() or predictions["meeting_date"].duplicated().any():
        raise ValueError("Prediction dates are missing or duplicated")
    probability_columns = ["probability_cut", "probability_hold", "probability_hike"]
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0):
        raise RuntimeError("Joint decision probabilities do not sum to one")
    return predictions.sort_values("meeting_date").reset_index(drop=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(_json_ready(payload), temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def train_models() -> tuple[Path, Path, Path]:
    """Train the hierarchy and write metrics, coefficients, and prediction audit."""
    panel = load_feature_panel()
    train, test = chronological_train_test_split(panel)
    X_train = train.loc[:, FEATURE_COLUMNS]
    X_test = test.loc[:, FEATURE_COLUMNS]

    primary_y = train[PRIMARY_TARGET].astype(int)
    primary = _tune_model(
        X_train,
        primary_y,
        class_weights=PRIMARY_CLASS_WEIGHT_OPTIONS,
        scoring=PRIMARY_MODEL_SCORING,
    )

    direction_train = train.loc[train[PRIMARY_TARGET].eq(1)].copy()
    direction_y = direction_train[DIAGNOSTIC_TARGET].astype(str)
    if set(direction_y) != {"cut", "hike"}:
        raise ValueError("Direction training rows must contain both cut and hike")
    direction_weights = [
        {"cut": float(weight), "hike": 1.0}
        for weight in DIRECTION_CUT_WEIGHT_OPTIONS
    ]
    direction = _tune_model(
        direction_train.loc[:, FEATURE_COLUMNS],
        direction_y,
        class_weights=direction_weights,
        scoring=DIAGNOSTIC_MODEL_SCORING,
    )

    primary, thresholds, policy_cv = _select_primary_model_and_policy(
        train,
        primary,
        direction.best_params_,
    )

    p_change = _positive_probability(primary, X_test, 1)
    p_cut_direction = _positive_probability(direction, X_test, "cut")
    holdout_policy = apply_decision_policy(p_change, p_cut_direction, thresholds)
    predictions = _build_prediction_table(test, holdout_policy)

    binary_probabilities = np.column_stack((1.0 - p_change, p_change))
    final_binary = predictions["predicted_is_change"].astype(int)
    raw_binary = predictions["raw_predicted_is_change"].astype(int)
    primary_metrics = _model_summary(
        primary,
        primary_y,
        target_name=PRIMARY_TARGET,
        scoring=DECISION_POLICY_SCORING,
    )
    primary_metrics["selection_note"] = (
        "C and class weight were selected jointly with the downstream policy "
        "on chronological training OOF predictions"
    )
    primary_metrics["holdout"] = _evaluate(
        test[PRIMARY_TARGET].astype(int),
        final_binary,
        binary_probabilities,
        [0, 1],
    )
    primary_metrics["raw_model_holdout"] = _evaluate(
        test[PRIMARY_TARGET].astype(int),
        raw_binary,
        binary_probabilities,
        [0, 1],
    )

    change_mask = test[PRIMARY_TARGET].eq(1).to_numpy()
    conditional_truth = test.loc[test[PRIMARY_TARGET].eq(1), DIAGNOSTIC_TARGET].astype(str)
    conditional_cut_probability = p_cut_direction[change_mask]
    conditional_prediction = np.where(
        conditional_cut_probability >= thresholds["direction_cut_threshold"],
        "cut",
        "hike",
    )
    conditional_probabilities = np.column_stack(
        (conditional_cut_probability, 1.0 - conditional_cut_probability)
    )
    direction_metrics = _model_summary(
        direction,
        direction_y,
        target_name=DIRECTION_TARGET,
        scoring=DIAGNOSTIC_MODEL_SCORING,
    )
    direction_metrics["definition"] = "cut versus hike, conditional on an actual change"
    direction_metrics["holdout"] = _evaluate(
        conditional_truth,
        pd.Series(conditional_prediction),
        conditional_probabilities,
        ["cut", "hike"],
    )

    joint_probabilities = predictions[
        ["probability_cut", "probability_hold", "probability_hike"]
    ].to_numpy()
    actual_decision = test[DIAGNOSTIC_TARGET].astype(str)
    decision_metrics = {
        "target": DIAGNOSTIC_TARGET,
        "architecture": "P(change) multiplied by P(cut or hike | change)",
        "thresholds": thresholds,
        "policy_cross_validation": policy_cv,
        "holdout": _evaluate(
            actual_decision,
            predictions["final_decision"].astype(str),
            joint_probabilities,
            list(DECISION_CLASSES),
        ),
        "raw_model_holdout": _evaluate(
            actual_decision,
            predictions["raw_model_decision"].astype(str),
            joint_probabilities,
            list(DECISION_CLASSES),
        ),
        "override_audit": {
            "obvious_cut_signal_count": int(
                predictions["obvious_cut_signal"].sum()
            ),
            "override_count": int(predictions["cut_override_triggered"].sum()),
            "correct_override_count": int(
                (
                    predictions["cut_override_triggered"]
                    & predictions["actual_decision"].eq("cut")
                ).sum()
            ),
            "incorrect_override_count": int(
                (
                    predictions["cut_override_triggered"]
                    & predictions["actual_decision"].ne("cut")
                ).sum()
            ),
        },
    }

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "feature_path": str(FEATURE_PANEL_PATH),
            "row_count": int(len(panel)),
            "feature_count": int(len(FEATURE_COLUMNS)),
            "first_meeting": panel.meeting_date.iloc[0].strftime("%Y-%m-%d"),
            "last_meeting": panel.meeting_date.iloc[-1].strftime("%Y-%m-%d"),
            "test_fraction": MODEL_TEST_FRACTION,
            "train": {
                "row_count": int(len(train)),
                "first_meeting": train.meeting_date.iloc[0].strftime("%Y-%m-%d"),
                "last_meeting": train.meeting_date.iloc[-1].strftime("%Y-%m-%d"),
            },
            "test": {
                "row_count": int(len(test)),
                "first_meeting": test.meeting_date.iloc[0].strftime("%Y-%m-%d"),
                "last_meeting": test.meeting_date.iloc[-1].strftime("%Y-%m-%d"),
            },
        },
        "methodology": {
            "holdout": "newest meetings; no shuffling; never used to select thresholds",
            "architecture": "hierarchical change model then cut-versus-hike model",
            "cross_validation": f"{CV_SPLITS}-split forward TimeSeriesSplit",
            "preprocessing": "StandardScaler fitted inside each model fit",
            "override": "three probability gates selected on training OOF predictions",
            "vintage_limit": (
                "macro dates precede meetings, but inputs are not unrevised "
                "point-in-time release vintages"
            ),
        },
        "models": {
            PRIMARY_TARGET: primary_metrics,
            DIRECTION_TARGET: direction_metrics,
            DIAGNOSTIC_TARGET: decision_metrics,
        },
    }
    coefficients = pd.concat(
        [
            _coefficients(primary, target_name=PRIMARY_TARGET),
            _coefficients(direction, target_name=DIRECTION_TARGET),
        ],
        ignore_index=True,
    )
    _atomic_json(MODEL_METRICS_PATH, metrics)
    _atomic_csv(MODEL_COEFFICIENTS_PATH, coefficients)
    _atomic_csv(MODEL_PREDICTIONS_PATH, predictions)
    return MODEL_METRICS_PATH, MODEL_COEFFICIENTS_PATH, MODEL_PREDICTIONS_PATH


def main() -> None:
    """Run model training using only configured project paths."""
    metrics_path, coefficients_path, predictions_path = train_models()
    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    binary = metrics["models"][PRIMARY_TARGET]["holdout"]
    decision = metrics["models"][DIAGNOSTIC_TARGET]["holdout"]
    override = metrics["models"][DIAGNOSTIC_TARGET]["override_audit"]
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved coefficients to {coefficients_path}")
    print(f"Saved holdout predictions to {predictions_path}")
    print(
        "Holdout change model: "
        f"accuracy={binary['accuracy']:.3f}, "
        f"balanced_accuracy={binary['balanced_accuracy']:.3f}, "
        f"macro_f1={binary['macro']['f1']:.3f}"
    )
    print(
        "Holdout decision policy: "
        f"accuracy={decision['accuracy']:.3f}, "
        f"balanced_accuracy={decision['balanced_accuracy']:.3f}, "
        f"macro_f1={decision['macro']['f1']:.3f}, "
        f"overrides={override['override_count']}"
    )


if __name__ == "__main__":
    main()
