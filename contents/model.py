"""Stage 4: train and evaluate Fed decision logistic-regression models.

Train two models from the same feature panel:

- primary: binary ``is_change`` (hold versus hike/cut)
- diagnostic: three-class ``decision`` (cut, hold, hike)

The scaler must be fitted inside each cross-validation fold, so it belongs in a
scikit-learn ``Pipeline`` with logistic regression.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    DECISION_CLASSES,
    DIAGNOSTIC_TARGET,
    FEATURE_COLUMNS,
    FEATURE_PANEL_PATH,
    CV_SPLITS,
    LOGISTIC_C_VALUES,
    LOGISTIC_MAX_ITERATIONS,
    MODEL_COEFFICIENTS_PATH,
    MODEL_METRICS_PATH,
    MODEL_PREDICTIONS_PATH,
    MODEL_TEST_FRACTION,
    PRIMARY_TARGET,
    PRIMARY_MODEL_SCORING,
    DIAGNOSTIC_MODEL_SCORING,
    RANDOM_STATE,
)
from features import validate_feature_panel


def _warning_free_balanced_accuracy(
    y_true: "pd.Series | np.ndarray",
    y_pred: "pd.Series | np.ndarray",
) -> float:
    """Compute mean recall over observed classes, including one-class folds."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    observed_classes = np.unique(true_values)
    if observed_classes.size == 0:
        raise ValueError("Balanced accuracy requires at least one observation")
    recalls = [
        np.mean(predicted_values[true_values == label] == label)
        for label in observed_classes
    ]
    return float(np.mean(recalls))


def _unwrap_pipeline(fitted_estimator: Any) -> Pipeline:
    """Return the fitted pipeline inside a search object or direct pipeline."""
    candidate = getattr(fitted_estimator, "best_estimator_", fitted_estimator)
    if not isinstance(candidate, Pipeline):
        raise TypeError("fitted_estimator must contain a fitted sklearn Pipeline")
    if "logistic_regression" not in candidate.named_steps:
        raise ValueError("Pipeline does not contain logistic_regression")
    classifier = candidate.named_steps["logistic_regression"]
    if not hasattr(classifier, "classes_"):
        raise ValueError("Logistic-regression pipeline has not been fitted")
    return candidate


def _json_value(value: Any) -> Any:
    """Convert a numpy scalar to its JSON-compatible Python equivalent."""
    return value.item() if isinstance(value, np.generic) else value


def load_feature_panel() -> "pd.DataFrame":
    """Load and validate the chronological feature panel.

    The input location is fixed by ``config.FEATURE_PANEL_PATH``.
    """
    if not FEATURE_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"Feature panel does not exist: {FEATURE_PANEL_PATH}. "
            "Run features.py first."
        )
    panel = pd.read_csv(FEATURE_PANEL_PATH)
    panel["meeting_date"] = pd.to_datetime(panel["meeting_date"], errors="coerce")
    validate_feature_panel(panel)
    return panel


def chronological_train_test_split(
    panel: "pd.DataFrame",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split older meetings into train and newer meetings into test.

    The holdout fraction comes from ``config.MODEL_TEST_FRACTION``. Both targets
    must have every declared class in both partitions so their holdout metrics
    remain meaningful. The test partition is never used during tuning.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    validate_feature_panel(panel)
    if not 0 < MODEL_TEST_FRACTION < 1:
        raise ValueError("MODEL_TEST_FRACTION must be strictly between 0 and 1")

    test_size = math.ceil(len(panel) * MODEL_TEST_FRACTION)
    split_position = len(panel) - test_size
    if split_position <= 0:
        raise ValueError("Feature panel is too small for the configured holdout")

    train = panel.iloc[:split_position].copy().reset_index(drop=True)
    test = panel.iloc[split_position:].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    if train["meeting_date"].max() >= test["meeting_date"].min():
        raise ValueError("Train/test dates overlap or are not strictly chronological")

    required_classes: dict[str, set[Any]] = {
        PRIMARY_TARGET: {0, 1},
        DIAGNOSTIC_TARGET: set(DECISION_CLASSES),
    }
    for target_name, expected in required_classes.items():
        for partition_name, partition in (("train", train), ("test", test)):
            observed = set(partition[target_name].tolist())
            missing = sorted(expected - observed, key=str)
            if missing:
                raise ValueError(
                    f"{partition_name} partition lacks {target_name} classes: {missing}"
                )
    return train, test


def make_estimator(*, multiclass: bool) -> "Pipeline":
    """Declare ``StandardScaler`` followed by ``LogisticRegression``.

    ``lbfgs`` supports binary and multinomial logistic regression. Modern
    scikit-learn selects the multiclass behavior from the observed target.
    """
    if not isinstance(multiclass, bool):
        raise TypeError("multiclass must be a boolean")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=LOGISTIC_MAX_ITERATIONS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_estimator(
    estimator: "Pipeline",
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    *,
    scoring: str,
) -> Any:
    """Tune logistic-regression C using chronological five-fold validation.

    Fold width is reduced when necessary so the earliest training prefix already
    contains every class observed in ``y_train``. This preserves five forward
    splits while avoiding invalid one-class logistic-regression fits.
    """
    if not isinstance(estimator, Pipeline):
        raise TypeError("estimator must be a scikit-learn Pipeline")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")
    if len(X_train) != len(y_train) or X_train.empty:
        raise ValueError("X_train and y_train must be non-empty and equally sized")
    if not isinstance(scoring, str) or not scoring.strip():
        raise ValueError("scoring must be a non-empty sklearn scoring name")

    numeric_X = X_train.apply(pd.to_numeric, errors="coerce")
    values = numeric_X.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("X_train contains missing or non-finite features")
    if y_train.isna().any():
        raise ValueError("y_train contains missing targets")

    required_classes = set(y_train.tolist())
    if len(required_classes) < 2:
        raise ValueError("y_train must contain at least two classes")
    minimum_initial_size: int | None = None
    for prefix_size in range(2, len(y_train) + 1):
        if set(y_train.iloc[:prefix_size].tolist()) == required_classes:
            minimum_initial_size = prefix_size
            break
    if minimum_initial_size is None:
        raise ValueError("Could not find a training prefix containing every class")

    maximum_test_size = (
        len(y_train) - minimum_initial_size
    ) // CV_SPLITS
    if maximum_test_size < 1:
        raise ValueError(
            "Training data is too short for chronological CV after all classes appear"
        )
    default_test_size = len(y_train) // (CV_SPLITS + 1)
    fold_test_size = min(default_test_size, maximum_test_size)
    cross_validation = TimeSeriesSplit(
        n_splits=CV_SPLITS,
        test_size=fold_test_size,
    )
    for train_indices, _ in cross_validation.split(numeric_X):
        observed = set(y_train.iloc[train_indices].tolist())
        if observed != required_classes:
            missing = sorted(required_classes - observed, key=str)
            raise ValueError(f"A CV training fold lacks target classes: {missing}")

    search_scoring: Any = scoring
    if scoring == "balanced_accuracy":
        search_scoring = make_scorer(_warning_free_balanced_accuracy)

    search = GridSearchCV(
        estimator=estimator,
        param_grid={"logistic_regression__C": list(LOGISTIC_C_VALUES)},
        scoring=search_scoring,
        cv=cross_validation,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(numeric_X, y_train)
    return search


def evaluate_classifier(
    fitted_estimator: Any,
    X_test: "pd.DataFrame",
    y_test: "pd.Series",
) -> dict[str, Any]:
    """Return JSON-serializable holdout metrics.

    Include the holdout date range outside this helper, where meeting dates are
    still available alongside the feature matrix.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series")
    if len(X_test) != len(y_test) or X_test.empty:
        raise ValueError("X_test and y_test must be non-empty and equally sized")
    values = X_test.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("X_test contains missing or non-finite features")
    if y_test.isna().any():
        raise ValueError("y_test contains missing targets")

    pipeline = _unwrap_pipeline(fitted_estimator)
    classifier = pipeline.named_steps["logistic_regression"]
    labels = [_json_value(value) for value in classifier.classes_]
    predictions = pipeline.predict(X_test)
    majority_class = y_test.value_counts().idxmax()
    majority_predictions = np.full(len(y_test), majority_class)

    class_counts = y_test.value_counts().reindex(classifier.classes_, fill_value=0)
    metrics: dict[str, Any] = {
        "sample_count": int(len(y_test)),
        "class_counts": {
            str(_json_value(label)): int(count)
            for label, count in class_counts.items()
        },
        "labels": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "majority_baseline": {
            "class": _json_value(majority_class),
            "accuracy": float(accuracy_score(y_test, majority_predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, majority_predictions)
            ),
        },
        "macro": {
            "precision": float(
                precision_score(
                    y_test, predictions, average="macro", zero_division=0
                )
            ),
            "recall": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
        },
        "weighted": {
            "precision": float(
                precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
        },
        "confusion_matrix": {
            "labels": labels,
            "values": confusion_matrix(
                y_test,
                predictions,
                labels=classifier.classes_,
            ).astype(int).tolist(),
        },
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        probability_metrics: dict[str, Any] = {
            "log_loss": float(
                log_loss(y_test, probabilities, labels=classifier.classes_)
            )
        }
        if len(classifier.classes_) == 2 and y_test.nunique() == 2:
            positive_class = classifier.classes_[1]
            binary_truth = y_test.eq(positive_class).astype(int)
            probability_metrics["roc_auc"] = float(
                roc_auc_score(binary_truth, probabilities[:, 1])
            )
            probability_metrics["brier_score"] = float(
                brier_score_loss(binary_truth, probabilities[:, 1])
            )
            probability_metrics["positive_class"] = _json_value(positive_class)
        elif (
            len(classifier.classes_) > 2
            and set(y_test.tolist()) == set(classifier.classes_.tolist())
        ):
            probability_metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=classifier.classes_,
                    multi_class="ovr",
                    average="macro",
                )
            )
        metrics["probability"] = probability_metrics
    return metrics


def extract_coefficients(
    fitted_estimator: Any,
    feature_names: tuple[str, ...],
    *,
    target_name: str,
) -> "pd.DataFrame":
    """Return tidy standardized coefficients for every class and feature.

    Binary sklearn models store one positive-class vector; this function emits
    that vector for the positive class and its sign-reversed equivalent for the
    negative class. Coefficients describe standardized association, not causation
    or standalone feature importance.
    """
    if not isinstance(feature_names, tuple) or not feature_names:
        raise ValueError("feature_names must be a non-empty tuple")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names contains duplicates")
    if target_name not in {PRIMARY_TARGET, DIAGNOSTIC_TARGET}:
        raise ValueError(f"Unknown target_name: {target_name}")

    pipeline = _unwrap_pipeline(fitted_estimator)
    classifier = pipeline.named_steps["logistic_regression"]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercepts = np.asarray(classifier.intercept_, dtype=float)
    classes = list(classifier.classes_)
    if coefficients.shape[1] != len(feature_names):
        raise ValueError(
            "Coefficient width does not match the supplied feature names"
        )

    if len(classes) == 2 and coefficients.shape[0] == 1:
        class_coefficients = np.vstack((-coefficients[0], coefficients[0]))
        class_intercepts = np.array((-intercepts[0], intercepts[0]))
    elif coefficients.shape[0] == len(classes):
        class_coefficients = coefficients
        class_intercepts = intercepts
    else:
        raise ValueError("Classifier classes and coefficient rows are inconsistent")

    rows: list[dict[str, Any]] = []
    for class_position, class_label in enumerate(classes):
        for feature_position, feature_name in enumerate(feature_names):
            coefficient = float(
                class_coefficients[class_position, feature_position]
            )
            rows.append(
                {
                    "target": target_name,
                    "model": "standard_scaler_logistic_regression",
                    "class": _json_value(class_label),
                    "feature": feature_name,
                    "standardized_coefficient": coefficient,
                    "odds_ratio_per_1sd": float(np.exp(coefficient)),
                    "intercept": float(class_intercepts[class_position]),
                    "regularization_C": float(classifier.C),
                }
            )
    return pd.DataFrame.from_records(rows)


def train_target(
    train: "pd.DataFrame",
    test: "pd.DataFrame",
    *,
    target_name: str,
) -> tuple[Any, dict[str, Any], "pd.DataFrame"]:
    """Train, tune, evaluate, and summarize one declared target.

    Only configured features are exposed to the estimator. Target-specific
    scoring names come from ``config.py``.
    """
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames")
    if train.empty or test.empty:
        raise ValueError("train and test must both contain observations")
    if target_name not in {PRIMARY_TARGET, DIAGNOSTIC_TARGET}:
        raise ValueError(f"Unknown target_name: {target_name}")

    required_columns = set(FEATURE_COLUMNS) | {target_name, "meeting_date"}
    for partition_name, partition in (("train", train), ("test", test)):
        missing = sorted(required_columns - set(partition.columns))
        if missing:
            raise ValueError(f"{partition_name} is missing columns: {missing}")
    if train["meeting_date"].max() >= test["meeting_date"].min():
        raise ValueError("train must end strictly before test begins")

    X_train = train.loc[:, FEATURE_COLUMNS]
    X_test = test.loc[:, FEATURE_COLUMNS]
    y_train = train[target_name]
    y_test = test[target_name]
    multiclass = target_name == DIAGNOSTIC_TARGET
    scoring = (
        DIAGNOSTIC_MODEL_SCORING if multiclass else PRIMARY_MODEL_SCORING
    )

    search = tune_estimator(
        make_estimator(multiclass=multiclass),
        X_train,
        y_train,
        scoring=scoring,
    )
    holdout_metrics = evaluate_classifier(search, X_test, y_test)
    train_counts = y_train.value_counts()
    metrics: dict[str, Any] = {
        "target": target_name,
        "scoring": scoring,
        "train_sample_count": int(len(train)),
        "train_class_counts": {
            str(_json_value(label)): int(count)
            for label, count in train_counts.items()
        },
        "cv_best_score": float(search.best_score_),
        "best_parameters": {
            key: _json_value(value) for key, value in search.best_params_.items()
        },
        "cv_splits": int(search.n_splits_),
        "holdout": holdout_metrics,
    }
    coefficients = extract_coefficients(
        search,
        FEATURE_COLUMNS,
        target_name=target_name,
    )
    return search, metrics, coefficients


def build_holdout_prediction_table(
    primary_estimator: Any,
    diagnostic_estimator: Any,
    test: "pd.DataFrame",
) -> "pd.DataFrame":
    """Return dated out-of-sample predictions for the chronological holdout."""
    if not isinstance(test, pd.DataFrame) or test.empty:
        raise ValueError("test must be a non-empty pandas DataFrame")
    required_columns = {
        "meeting_date",
        *FEATURE_COLUMNS,
        PRIMARY_TARGET,
        DIAGNOSTIC_TARGET,
    }
    missing = sorted(required_columns - set(test.columns))
    if missing:
        raise ValueError(f"test is missing prediction-table columns: {missing}")

    primary_pipeline = _unwrap_pipeline(primary_estimator)
    diagnostic_pipeline = _unwrap_pipeline(diagnostic_estimator)
    primary_classes = list(
        primary_pipeline.named_steps["logistic_regression"].classes_
    )
    diagnostic_classes = list(
        diagnostic_pipeline.named_steps["logistic_regression"].classes_
    )
    if set(primary_classes) != {0, 1}:
        raise ValueError(f"Primary model classes must be 0 and 1: {primary_classes}")
    if set(diagnostic_classes) != set(DECISION_CLASSES):
        raise ValueError(
            "Diagnostic model classes disagree with configured decisions: "
            f"{diagnostic_classes}"
        )

    X_test = test.loc[:, FEATURE_COLUMNS]
    primary_probabilities = primary_pipeline.predict_proba(X_test)
    diagnostic_probabilities = diagnostic_pipeline.predict_proba(X_test)
    if not np.allclose(primary_probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("Primary prediction probabilities do not sum to one")
    if not np.allclose(diagnostic_probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("Diagnostic prediction probabilities do not sum to one")

    primary_positions = {
        _json_value(label): position for position, label in enumerate(primary_classes)
    }
    diagnostic_positions = {
        str(_json_value(label)): position
        for position, label in enumerate(diagnostic_classes)
    }
    predictions = pd.DataFrame(
        {
            "meeting_date": pd.to_datetime(test["meeting_date"], errors="coerce"),
            "dataset_split": "chronological_holdout",
            "actual_is_change": test[PRIMARY_TARGET].astype("int8").to_numpy(),
            "predicted_is_change": primary_pipeline.predict(X_test).astype("int8"),
            "probability_change": primary_probabilities[
                :, primary_positions[1]
            ],
            "actual_decision": test[DIAGNOSTIC_TARGET].astype("string").to_numpy(),
            "predicted_decision": diagnostic_pipeline.predict(X_test),
            "probability_cut": diagnostic_probabilities[
                :, diagnostic_positions["cut"]
            ],
            "probability_hold": diagnostic_probabilities[
                :, diagnostic_positions["hold"]
            ],
            "probability_hike": diagnostic_probabilities[
                :, diagnostic_positions["hike"]
            ],
        }
    )
    if predictions["meeting_date"].isna().any():
        raise ValueError("test contains missing or invalid meeting dates")
    if predictions["meeting_date"].duplicated().any():
        raise ValueError("test contains duplicate meeting dates")
    if not predictions["actual_decision"].isin(DECISION_CLASSES).all():
        raise ValueError("Prediction table contains invalid actual decisions")
    if not predictions["predicted_decision"].isin(DECISION_CLASSES).all():
        raise ValueError("Prediction table contains invalid predicted decisions")
    if not predictions["actual_is_change"].isin([0, 1]).all():
        raise ValueError("Prediction table contains invalid actual binary targets")
    if not predictions["predicted_is_change"].isin([0, 1]).all():
        raise ValueError("Prediction table contains invalid binary predictions")
    return predictions.sort_values("meeting_date").reset_index(drop=True)


def train_models() -> tuple[Path, Path, Path]:
    """Train both targets and write metrics, coefficients, and predictions.

    All paths and the holdout fraction come from ``config.py``. Estimators are
    deliberately not serialized; the JSON and CSV artifacts are safe to inspect
    and the full training run is inexpensive and reproducible.
    """
    panel = load_feature_panel()
    train, test = chronological_train_test_split(panel)

    primary_estimator, primary_metrics, primary_coefficients = train_target(
        train,
        test,
        target_name=PRIMARY_TARGET,
    )
    diagnostic_estimator, diagnostic_metrics, diagnostic_coefficients = train_target(
        train,
        test,
        target_name=DIAGNOSTIC_TARGET,
    )

    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "feature_path": str(FEATURE_PANEL_PATH),
            "row_count": int(len(panel)),
            "feature_count": int(len(FEATURE_COLUMNS)),
            "first_meeting": panel["meeting_date"].iloc[0].strftime("%Y-%m-%d"),
            "last_meeting": panel["meeting_date"].iloc[-1].strftime("%Y-%m-%d"),
            "test_fraction": MODEL_TEST_FRACTION,
            "train": {
                "row_count": int(len(train)),
                "first_meeting": train["meeting_date"].iloc[0].strftime("%Y-%m-%d"),
                "last_meeting": train["meeting_date"].iloc[-1].strftime("%Y-%m-%d"),
            },
            "test": {
                "row_count": int(len(test)),
                "first_meeting": test["meeting_date"].iloc[0].strftime("%Y-%m-%d"),
                "last_meeting": test["meeting_date"].iloc[-1].strftime("%Y-%m-%d"),
            },
        },
        "methodology": {
            "holdout": "newest meetings; no shuffling",
            "cross_validation": (
                f"{CV_SPLITS}-split TimeSeriesSplit with class-complete "
                "training prefixes"
            ),
            "preprocessing": "StandardScaler fitted inside each CV pipeline fold",
            "coefficient_interpretation": (
                "standardized association; not causation or standalone importance"
            ),
            "vintage_limit": (
                "macro reference periods are aligned before meetings, but the "
                "panel is not based on unrevised point-in-time release vintages"
            ),
        },
        "models": {
            PRIMARY_TARGET: primary_metrics,
            DIAGNOSTIC_TARGET: diagnostic_metrics,
        },
    }
    coefficients = pd.concat(
        [primary_coefficients, diagnostic_coefficients],
        ignore_index=True,
    )
    predictions = build_holdout_prediction_table(
        primary_estimator,
        diagnostic_estimator,
        test,
    )

    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{MODEL_METRICS_PATH.name}.",
            suffix=".tmp",
            dir=MODEL_METRICS_PATH.parent,
            delete=False,
        ) as temporary_file:
            metrics_temporary_path = Path(temporary_file.name)
            json.dump(metrics, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        metrics_temporary_path.replace(MODEL_METRICS_PATH)
    except Exception:
        if metrics_temporary_path is not None:
            metrics_temporary_path.unlink(missing_ok=True)
        raise

    coefficients_temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{MODEL_COEFFICIENTS_PATH.name}.",
            suffix=".tmp",
            dir=MODEL_COEFFICIENTS_PATH.parent,
            delete=False,
        ) as temporary_file:
            coefficients_temporary_path = Path(temporary_file.name)
            coefficients.to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        coefficients_temporary_path.replace(MODEL_COEFFICIENTS_PATH)
    except Exception:
        if coefficients_temporary_path is not None:
            coefficients_temporary_path.unlink(missing_ok=True)
        raise

    predictions_temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{MODEL_PREDICTIONS_PATH.name}.",
            suffix=".tmp",
            dir=MODEL_PREDICTIONS_PATH.parent,
            delete=False,
        ) as temporary_file:
            predictions_temporary_path = Path(temporary_file.name)
            predictions.to_csv(
                temporary_file,
                index=False,
                date_format="%Y-%m-%d",
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        predictions_temporary_path.replace(MODEL_PREDICTIONS_PATH)
    except Exception:
        if predictions_temporary_path is not None:
            predictions_temporary_path.unlink(missing_ok=True)
        raise
    return MODEL_METRICS_PATH, MODEL_COEFFICIENTS_PATH, MODEL_PREDICTIONS_PATH


def main() -> None:
    """Train both configured targets and report the primary holdout result."""
    metrics_path, coefficients_path, predictions_path = train_models()
    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    primary_holdout = metrics["models"][PRIMARY_TARGET]["holdout"]

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved coefficients to {coefficients_path}")
    print(f"Saved holdout predictions to {predictions_path}")
    print(
        "Primary holdout: "
        f"accuracy={primary_holdout['accuracy']:.3f}, "
        f"balanced_accuracy={primary_holdout['balanced_accuracy']:.3f}, "
        f"macro_f1={primary_holdout['macro']['f1']:.3f}, "
        f"roc_auc={primary_holdout['probability']['roc_auc']:.3f}"
    )
    print(
        "Majority baseline: "
        f"accuracy={primary_holdout['majority_baseline']['accuracy']:.3f}, "
        "balanced_accuracy="
        f"{primary_holdout['majority_baseline']['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
