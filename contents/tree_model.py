"""Train decision-tree and random-forest FOMC decision classifiers.

This is a standalone comparison model. It consumes the same validated feature
panel and chronological holdout as ``model.py`` but predicts the three decision
classes directly. Hyperparameters and the winning model family are selected
only with forward cross-validation inside the training period.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from config import (
    CV_SPLITS,
    DECISION_CLASSES,
    DIAGNOSTIC_TARGET,
    FEATURE_COLUMNS,
    MODEL_TEST_FRACTION,
    RANDOM_STATE,
    TREE_MODEL_FACTOR_RANKINGS_PATH,
    TREE_MODEL_IMPORTANCE_PATH,
    TREE_MODEL_METRICS_PATH,
    TREE_MODEL_PREDICTIONS_PATH,
)
from model import chronological_train_test_split, load_feature_panel


FACTOR_RANKING_LIMIT = 10


def build_chronological_cv(target: pd.Series) -> TimeSeriesSplit:
    """Return forward folds whose training prefix contains every decision."""
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if len(target) < CV_SPLITS + 2:
        raise ValueError("target is too short for configured cross-validation")

    target = target.reset_index(drop=True).astype("string")
    required_classes = set(DECISION_CLASSES)
    if set(target) != required_classes:
        raise ValueError("target must contain cut, hold, and hike")

    first_complete: int | None = None
    for prefix_size in range(2, len(target) + 1):
        if set(target.iloc[:prefix_size]) == required_classes:
            first_complete = prefix_size
            break
    if first_complete is None:
        raise ValueError("no chronological training prefix contains every class")

    maximum_test_size = (len(target) - first_complete) // CV_SPLITS
    if maximum_test_size < 1:
        raise ValueError("not enough rows for class-complete forward folds")
    test_size = min(len(target) // (CV_SPLITS + 1), maximum_test_size)
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=test_size)

    for train_index, validation_index in splitter.split(target):
        if set(target.iloc[train_index]) != required_classes:
            raise ValueError("a chronological training fold lacks a decision class")
        if train_index.max() >= validation_index.min():
            raise RuntimeError("cross-validation fold is not chronological")
    return splitter


def tune_decision_tree(
    features: pd.DataFrame,
    target: pd.Series,
    cv: TimeSeriesSplit,
) -> GridSearchCV:
    """Tune a single decision tree with forward cross-validation."""
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 3, 5, 10],
            "class_weight": [None, "balanced"],
        },
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(features, target)
    return search


def tune_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    cv: TimeSeriesSplit,
) -> GridSearchCV:
    """Tune a random forest with forward cross-validation."""
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        param_grid={
            "max_depth": [5, 10, None],
            "min_samples_leaf": [1, 3, 5],
            "max_features": ["sqrt", 0.5],
            "class_weight": [None, "balanced"],
        },
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(features, target)
    return search


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Calculate overall and per-class holdout metrics."""
    labels = list(DECISION_CLASSES)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=labels
        ).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def build_prediction_table(
    test: pd.DataFrame,
    searches: dict[str, GridSearchCV],
    selected_model: str,
) -> pd.DataFrame:
    """Return predictions and class probabilities for both model families."""
    features = test.loc[:, FEATURE_COLUMNS]
    predictions = pd.DataFrame(
        {
            "meeting_date": test["meeting_date"].to_numpy(),
            "actual_decision": test[DIAGNOSTIC_TARGET].astype(str).to_numpy(),
        }
    )

    for model_name, search in searches.items():
        estimator = search.best_estimator_
        model_prediction = estimator.predict(features)
        model_probabilities = estimator.predict_proba(features)
        class_positions = {
            label: index for index, label in enumerate(estimator.classes_)
        }
        predictions[f"{model_name}_prediction"] = model_prediction
        predictions[f"{model_name}_correct"] = (
            model_prediction == predictions["actual_decision"]
        )
        for label in DECISION_CLASSES:
            predictions[f"{model_name}_probability_{label}"] = (
                model_probabilities[:, class_positions[label]]
            )

    predictions["selected_model"] = selected_model
    predictions["selected_prediction"] = predictions[
        f"{selected_model}_prediction"
    ]
    predictions["selected_correct"] = predictions[
        f"{selected_model}_correct"
    ]
    return predictions


def build_feature_importance_table(
    searches: dict[str, GridSearchCV],
) -> pd.DataFrame:
    """Return impurity-based feature importances for both fitted estimators."""
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS})
    for model_name, search in searches.items():
        values = search.best_estimator_.feature_importances_
        importance[f"{model_name}_importance"] = values
        importance[f"{model_name}_rank"] = (
            pd.Series(values).rank(method="min", ascending=False).astype(int)
        )
    return importance.sort_values(
        ["random_forest_rank", "decision_tree_rank", "feature"]
    ).reset_index(drop=True)


def build_factor_ranking_table(
    importance: pd.DataFrame,
) -> pd.DataFrame:
    """List the most and least influential features for both model families.

    Influence is the fitted estimator's impurity-based ``feature_importances_``
    value. Zero-importance features are omitted from the most-influential list
    but retained in the least-influential list. Alphabetical order breaks ties.
    """
    records: list[dict[str, Any]] = []
    for model_name in ("decision_tree", "random_forest"):
        importance_column = f"{model_name}_importance"
        if importance_column not in importance.columns:
            raise ValueError(f"importance is missing {importance_column}")

        ordered_most = importance.sort_values(
            [importance_column, "feature"],
            ascending=[False, True],
        )
        ordered_most = ordered_most.loc[
            ordered_most[importance_column] > 0
        ].head(FACTOR_RANKING_LIMIT)
        ordered_least = importance.sort_values(
            [importance_column, "feature"],
            ascending=[True, True],
        ).head(FACTOR_RANKING_LIMIT)

        for influence_group, ranked_rows in (
            ("most_influential", ordered_most),
            ("least_influential", ordered_least),
        ):
            for rank, (_, row) in enumerate(ranked_rows.iterrows(), start=1):
                records.append(
                    {
                        "model": model_name,
                        "influence_group": influence_group,
                        "rank": rank,
                        "feature": row["feature"],
                        "importance": float(row[importance_column]),
                    }
                )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "model",
            "influence_group",
            "rank",
            "feature",
            "importance",
        ],
    )


def factor_rankings_for_json(rankings: pd.DataFrame) -> dict[str, Any]:
    """Convert the ranking table into model/group lists for metrics JSON."""
    result: dict[str, Any] = {}
    for model_name in ("decision_tree", "random_forest"):
        result[model_name] = {}
        model_rows = rankings.loc[rankings["model"].eq(model_name)]
        for influence_group in ("most_influential", "least_influential"):
            group_rows = model_rows.loc[
                model_rows["influence_group"].eq(influence_group)
            ].sort_values("rank")
            result[model_name][influence_group] = [
                {
                    "rank": int(row["rank"]),
                    "feature": str(row["feature"]),
                    "importance": float(row["importance"]),
                }
                for _, row in group_rows.iterrows()
            ]
    return result


def _json_ready(value: Any) -> Any:
    """Convert numpy values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically inside the configured output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(_json_ready(payload), temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame atomically inside the output directory."""
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


def train_tree_models() -> tuple[Path, Path, Path, Path]:
    """Train, compare, evaluate, and save both tree-based classifiers."""
    panel = load_feature_panel()
    train, test = chronological_train_test_split(panel)
    training_features = train.loc[:, FEATURE_COLUMNS]
    training_target = train[DIAGNOSTIC_TARGET].astype(str)
    test_target = test[DIAGNOSTIC_TARGET].astype(str)
    cv = build_chronological_cv(training_target)

    searches = {
        "decision_tree": tune_decision_tree(
            training_features, training_target, cv
        ),
        "random_forest": tune_random_forest(
            training_features, training_target, cv
        ),
    }
    selected_model = max(
        searches,
        key=lambda name: (float(searches[name].best_score_), name == "random_forest"),
    )

    model_metrics: dict[str, Any] = {}
    for model_name, search in searches.items():
        holdout_prediction = search.best_estimator_.predict(
            test.loc[:, FEATURE_COLUMNS]
        )
        model_metrics[model_name] = {
            "best_cv_macro_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
            "holdout": calculate_metrics(test_target, holdout_prediction),
        }

    importance = build_feature_importance_table(searches)
    factor_rankings = build_factor_ranking_table(importance)
    metrics = {
        "methodology": {
            "split": (
                f"oldest {(1 - MODEL_TEST_FRACTION):.0%} train, newest "
                f"{MODEL_TEST_FRACTION:.0%} untouched holdout"
            ),
            "cross_validation": f"{CV_SPLITS}-split forward TimeSeriesSplit",
            "selection_metric": "training cross-validated macro F1",
            "target": DIAGNOSTIC_TARGET,
        },
        "dataset": {
            "feature_count": len(FEATURE_COLUMNS),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_first_meeting": train["meeting_date"].min().date().isoformat(),
            "train_last_meeting": train["meeting_date"].max().date().isoformat(),
            "test_first_meeting": test["meeting_date"].min().date().isoformat(),
            "test_last_meeting": test["meeting_date"].max().date().isoformat(),
        },
        "selected_model": selected_model,
        "models": model_metrics,
        "feature_influence": {
            "method": "mean decrease in impurity (feature_importances_)",
            "caution": (
                "Correlated features can divide or exchange importance; zero "
                "importance in one fitted tree is not proof of no economic value."
            ),
            "ranking_limit_per_group": FACTOR_RANKING_LIMIT,
            "rankings": factor_rankings_for_json(factor_rankings),
        },
    }
    predictions = build_prediction_table(test, searches, selected_model)

    _atomic_write_json(metrics, TREE_MODEL_METRICS_PATH)
    _atomic_write_csv(predictions, TREE_MODEL_PREDICTIONS_PATH)
    _atomic_write_csv(importance, TREE_MODEL_IMPORTANCE_PATH)
    _atomic_write_csv(factor_rankings, TREE_MODEL_FACTOR_RANKINGS_PATH)
    return (
        TREE_MODEL_METRICS_PATH,
        TREE_MODEL_PREDICTIONS_PATH,
        TREE_MODEL_IMPORTANCE_PATH,
        TREE_MODEL_FACTOR_RANKINGS_PATH,
    )


def main() -> None:
    """Train both tree models and report their holdout results."""
    (
        metrics_path,
        predictions_path,
        importance_path,
        factor_rankings_path,
    ) = train_tree_models()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"Selected by training CV: {metrics['selected_model']}")
    for model_name, model_result in metrics["models"].items():
        holdout = model_result["holdout"]
        print(
            f"Test split: {(1 - MODEL_TEST_FRACTION)*100:.0f}/{MODEL_TEST_FRACTION*100:.0f}\n"
            f"{model_name}: cv_macro_f1={model_result['best_cv_macro_f1']:.3f}, "
            f"holdout_accuracy={holdout['accuracy']:.3f}, "
            f"holdout_balanced_accuracy={holdout['balanced_accuracy']:.3f}, "
            f"holdout_macro_f1={holdout['macro_f1']:.3f}"
        )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved feature importances to {importance_path}")
    print(f"Saved most/least influential factors to {factor_rankings_path}")


if __name__ == "__main__":
    main()
