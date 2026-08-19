"""Train decision-tree and random-forest FOMC decision classifiers.

This is a standalone comparison model. It consumes the same validated feature
panel and chronological holdout as ``model.py``. Each tree family first predicts
hold versus change, then conditionally predicts cut versus hike. Hyperparameters,
decision thresholds, and the winning family are selected only with forward
cross-validation inside the training period.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from config import (
    CV_SPLITS,
    DECISION_CLASSES,
    DIAGNOSTIC_TARGET,
    DIAGNOSTIC_MODEL_SCORING,
    DIRECTION_CUT_WEIGHT_OPTIONS,
    FEATURE_COLUMNS,
    MODEL_TEST_FRACTION,
    PRIMARY_CLASS_WEIGHT_OPTIONS,
    PRIMARY_MODEL_SCORING,
    PRIMARY_TARGET,
    RANDOM_STATE,
    TREE_MODEL_FACTOR_RANKINGS_PATH,
    TREE_MODEL_IMPORTANCE_PATH,
    TREE_MODEL_METRICS_PATH,
    TREE_MODEL_PREDICTIONS_PATH,
)
from model import (
    apply_decision_policy,
    chronological_train_test_split,
    load_feature_panel,
    select_decision_policy,
)


FACTOR_RANKING_LIMIT = 10


def build_chronological_cv(target: pd.Series) -> TimeSeriesSplit:
    """Return forward folds whose training prefix contains every target class."""
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if len(target) < CV_SPLITS + 2:
        raise ValueError("target is too short for configured cross-validation")

    target = target.reset_index(drop=True)
    required_classes = set(target.tolist())
    if len(required_classes) < 2:
        raise ValueError("target must contain at least two classes")

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


def _observed_class_balanced_accuracy(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    """Calculate mean recall over classes observed in one validation fold."""
    truth = np.asarray(actual)
    estimate = np.asarray(predicted)
    return float(
        np.mean(
            [
                np.mean(estimate[truth == label] == label)
                for label in np.unique(truth)
            ]
        )
    )


def _scorer(scoring: str) -> Any:
    """Return the configured scorer without one-class-fold warnings."""
    if scoring == "balanced_accuracy":
        return make_scorer(_observed_class_balanced_accuracy)
    return scoring


def tune_decision_tree(
    features: pd.DataFrame,
    target: pd.Series,
    cv: TimeSeriesSplit,
    *,
    class_weights: list[Any],
    scoring: str,
) -> GridSearchCV:
    """Tune one decision-tree hierarchy component with forward CV."""
    search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 3, 5, 10],
            "class_weight": class_weights,
        },
        scoring=_scorer(scoring),
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
    *,
    class_weights: list[Any],
    scoring: str,
) -> GridSearchCV:
    """Tune one random-forest hierarchy component with forward CV."""
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
            "class_weight": class_weights,
        },
        scoring=_scorer(scoring),
        cv=cv,
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(features, target)
    return search


def _positive_probability(
    estimator: Any,
    features: pd.DataFrame,
    positive_class: Any,
) -> np.ndarray:
    """Return a fitted classifier's probability for one named class."""
    classes = list(estimator.classes_)
    if positive_class not in classes:
        raise ValueError(
            f"estimator lacks positive class {positive_class!r}: {classes}"
        )
    return estimator.predict_proba(features)[:, classes.index(positive_class)]


def build_hierarchical_oof_probabilities(
    train: pd.DataFrame,
    component_searches: dict[str, GridSearchCV],
) -> pd.DataFrame:
    """Create forward out-of-fold probabilities for policy selection."""
    required_components = {"change", "direction"}
    if set(component_searches) != required_components:
        raise ValueError(
            f"component searches must be exactly {sorted(required_components)}"
        )

    features = train.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    change_target = train[PRIMARY_TARGET].astype(int).reset_index(drop=True)
    decision_target = train[DIAGNOSTIC_TARGET].astype(str).reset_index(drop=True)
    probabilities = pd.DataFrame(
        index=train.index,
        columns=["probability_change", "probability_cut_given_change"],
        dtype=float,
    )
    policy_cv = build_chronological_cv(decision_target)

    for fit_index, validation_index in policy_cv.split(features):
        change_estimator = clone(
            component_searches["change"].best_estimator_
        )
        change_estimator.fit(
            features.iloc[fit_index], change_target.iloc[fit_index]
        )
        probabilities.loc[validation_index, "probability_change"] = (
            _positive_probability(
                change_estimator,
                features.iloc[validation_index],
                1,
            )
        )

        direction_fit_index = fit_index[
            change_target.iloc[fit_index].to_numpy() == 1
        ]
        direction_target = decision_target.iloc[direction_fit_index]
        if set(direction_target) != {"cut", "hike"}:
            raise ValueError("an OOF direction-training fold lacks cut or hike")
        direction_estimator = clone(
            component_searches["direction"].best_estimator_
        )
        direction_estimator.fit(
            features.iloc[direction_fit_index], direction_target
        )
        probabilities.loc[
            validation_index, "probability_cut_given_change"
        ] = _positive_probability(
            direction_estimator,
            features.iloc[validation_index],
            "cut",
        )

    complete = probabilities.dropna().copy()
    if complete.empty:
        raise ValueError("chronological CV produced no policy probabilities")
    complete.insert(
        0,
        "actual_decision",
        decision_target.loc[complete.index].to_numpy(),
    )
    return complete.reset_index(drop=True)


def apply_hierarchical_tree_policy(
    test: pd.DataFrame,
    component_searches: dict[str, GridSearchCV],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Apply fitted hold/change and cut/hike trees to one test frame."""
    features = test.loc[:, FEATURE_COLUMNS]
    probability_change = _positive_probability(
        component_searches["change"].best_estimator_, features, 1
    )
    probability_cut_given_change = _positive_probability(
        component_searches["direction"].best_estimator_, features, "cut"
    )
    return apply_decision_policy(
        probability_change,
        probability_cut_given_change,
        thresholds,
    )


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
    policies: dict[str, pd.DataFrame],
    selected_model: str,
) -> pd.DataFrame:
    """Return hierarchical predictions and probabilities for both families."""
    predictions = pd.DataFrame(
        {
            "meeting_date": test["meeting_date"].to_numpy(),
            "actual_decision": test[DIAGNOSTIC_TARGET].astype(str).to_numpy(),
        }
    )

    policy_columns = (
        "raw_model_decision",
        "final_decision",
        "probability_change",
        "probability_cut_given_change",
        "probability_cut",
        "probability_hold",
        "probability_hike",
        "obvious_cut_signal",
        "cut_override_triggered",
        "override_reason",
    )
    for model_name, policy in policies.items():
        policy = policy.reset_index(drop=True)
        if len(policy) != len(test):
            raise ValueError(f"{model_name} policy length disagrees with test data")
        predictions[f"{model_name}_raw_prediction"] = policy[
            "raw_model_decision"
        ]
        predictions[f"{model_name}_prediction"] = policy["final_decision"]
        predictions[f"{model_name}_correct"] = (
            predictions[f"{model_name}_prediction"]
            == predictions["actual_decision"]
        )
        for policy_column in policy_columns[2:]:
            predictions[f"{model_name}_{policy_column}"] = policy[policy_column]

    predictions["selected_model"] = selected_model
    predictions["selected_prediction"] = predictions[
        f"{selected_model}_prediction"
    ]
    predictions["selected_correct"] = predictions[
        f"{selected_model}_correct"
    ]
    return predictions


def build_feature_importance_table(
    searches: dict[str, dict[str, GridSearchCV]],
) -> pd.DataFrame:
    """Return component and combined importances for both tree families."""
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS})
    for model_name, component_searches in searches.items():
        change_values = (
            component_searches["change"].best_estimator_.feature_importances_
        )
        direction_values = (
            component_searches["direction"].best_estimator_.feature_importances_
        )
        combined_values = (change_values + direction_values) / 2.0
        importance[f"{model_name}_change_importance"] = change_values
        importance[f"{model_name}_direction_importance"] = direction_values
        importance[f"{model_name}_importance"] = combined_values
        importance[f"{model_name}_rank"] = (
            pd.Series(combined_values)
            .rank(method="min", ascending=False)
            .astype(int)
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
    """Train, compare, evaluate, and save both hierarchical tree families."""
    panel = load_feature_panel()
    train, test = chronological_train_test_split(panel)
    training_features = train.loc[:, FEATURE_COLUMNS]
    change_target = train[PRIMARY_TARGET].astype(int)
    direction_train = train.loc[train[PRIMARY_TARGET].eq(1)].copy()
    direction_target = direction_train[DIAGNOSTIC_TARGET].astype(str)
    test_target = test[DIAGNOSTIC_TARGET].astype(str)
    if set(direction_target) != {"cut", "hike"}:
        raise ValueError("direction training rows must contain cut and hike")

    change_cv = build_chronological_cv(change_target)
    direction_cv = build_chronological_cv(direction_target)
    primary_weights = list(PRIMARY_CLASS_WEIGHT_OPTIONS)
    direction_weights = [
        {"cut": float(weight), "hike": 1.0}
        for weight in DIRECTION_CUT_WEIGHT_OPTIONS
    ]
    searches: dict[str, dict[str, GridSearchCV]] = {
        "decision_tree": {
            "change": tune_decision_tree(
                training_features,
                change_target,
                change_cv,
                class_weights=primary_weights,
                scoring=PRIMARY_MODEL_SCORING,
            ),
            "direction": tune_decision_tree(
                direction_train.loc[:, FEATURE_COLUMNS],
                direction_target,
                direction_cv,
                class_weights=direction_weights,
                scoring=DIAGNOSTIC_MODEL_SCORING,
            ),
        },
        "random_forest": {
            "change": tune_random_forest(
                training_features,
                change_target,
                change_cv,
                class_weights=primary_weights,
                scoring=PRIMARY_MODEL_SCORING,
            ),
            "direction": tune_random_forest(
                direction_train.loc[:, FEATURE_COLUMNS],
                direction_target,
                direction_cv,
                class_weights=direction_weights,
                scoring=DIAGNOSTIC_MODEL_SCORING,
            ),
        },
    }

    policies: dict[str, pd.DataFrame] = {}
    thresholds_by_model: dict[str, dict[str, float]] = {}
    policy_audits: dict[str, dict[str, Any]] = {}
    for model_name, component_searches in searches.items():
        oof = build_hierarchical_oof_probabilities(train, component_searches)
        thresholds, audit = select_decision_policy(oof)
        thresholds_by_model[model_name] = thresholds
        policy_audits[model_name] = audit
        policies[model_name] = apply_hierarchical_tree_policy(
            test,
            component_searches,
            thresholds,
        )

    selected_model = max(
        searches,
        key=lambda name: (
            float(policy_audits[name]["macro_f1"]),
            name == "random_forest",
        ),
    )

    model_metrics: dict[str, Any] = {}
    for model_name, component_searches in searches.items():
        policy = policies[model_name]
        model_metrics[model_name] = {
            "architecture": "hold/change then conditional cut/hike",
            "components": {
                "change": {
                    "cv_score": float(component_searches["change"].best_score_),
                    "scoring": PRIMARY_MODEL_SCORING,
                    "best_parameters": component_searches["change"].best_params_,
                },
                "direction": {
                    "cv_score": float(
                        component_searches["direction"].best_score_
                    ),
                    "scoring": DIAGNOSTIC_MODEL_SCORING,
                    "best_parameters": component_searches[
                        "direction"
                    ].best_params_,
                },
            },
            "decision_policy": {
                "thresholds": thresholds_by_model[model_name],
                "training_oof_audit": policy_audits[model_name],
            },
            "holdout": calculate_metrics(
                test_target,
                policy["final_decision"].to_numpy(),
            ),
            "holdout_override_audit": {
                "obvious_cut_signal_count": int(
                    policy["obvious_cut_signal"].sum()
                ),
                "override_count": int(policy["cut_override_triggered"].sum()),
            },
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
            "architecture": "hold/change then conditional cut/hike",
            "selection_metric": "training OOF three-class policy macro F1",
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
            "method": (
                "equal mean of hold/change and conditional cut/hike mean "
                "decrease in impurity (feature_importances_)"
            ),
            "caution": (
                "Correlated features can divide or exchange importance; zero "
                "importance in one fitted tree is not proof of no economic value."
            ),
            "ranking_limit_per_group": FACTOR_RANKING_LIMIT,
            "rankings": factor_rankings_for_json(factor_rankings),
        },
    }
    predictions = build_prediction_table(test, policies, selected_model)

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
        policy_cv_macro_f1 = model_result["decision_policy"][
            "training_oof_audit"
        ]["macro_f1"]
        print(
            f"Test split: {(1 - MODEL_TEST_FRACTION)*100:.0f}/{MODEL_TEST_FRACTION*100:.0f}\n"
            f"{model_name}: policy_cv_macro_f1={policy_cv_macro_f1:.3f}, "
            f"holdout_accuracy={holdout['accuracy']:.3f}, "
            f"holdout_balanced_accuracy={holdout['balanced_accuracy']:.3f}, "
            f"holdout_macro_f1={holdout['macro_f1']:.3f}, "
            f"overrides={model_result['holdout_override_audit']['override_count']}"
        )
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved feature importances to {importance_path}")
    print(f"Saved most/least influential factors to {factor_rankings_path}")


if __name__ == "__main__":
    main()
