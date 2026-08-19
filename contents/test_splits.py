"""Measure random-forest sensitivity to random states 0 through 100.

Random-forest hyperparameters are selected once with forward cross-validation
on the training period, then held fixed while only ``random_state`` changes.
The final holdout is reported for diagnosis, including every meeting-level
prediction, but must not be used to select the seed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_validate

from config import (
    CV_SPLITS,
    DECISION_CLASSES,
    DIAGNOSTIC_TARGET,
    FEATURE_COLUMNS,
    OUTPUTS,
    RANDOM_STATE,
)
from tree_model import (
    build_chronological_cv,
    chronological_train_test_split,
    load_feature_panel,
    tune_random_forest,
)


FIRST_RANDOM_STATE = 0
LAST_RANDOM_STATE = 100
RESULTS_PATH = OUTPUTS / "random_state_cv_results.csv"
HOLDOUT_PREDICTIONS_PATH = OUTPUTS / "random_state_holdout_predictions.csv"


def evaluate_random_states() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return CV metrics and explicit holdout predictions for every seed."""
    panel = load_feature_panel()
    train, holdout = chronological_train_test_split(panel)
    training_features = train.loc[:, FEATURE_COLUMNS]
    training_target = train[DIAGNOSTIC_TARGET].astype(str)
    holdout_features = holdout.loc[:, FEATURE_COLUMNS]
    holdout_target = holdout[DIAGNOSTIC_TARGET].astype(str)
    cv = build_chronological_cv(training_target)

    # Select the forest structure once. Re-running the parameter search for
    # every seed would confound seed sensitivity with hyperparameter changes.
    tuned_search = tune_random_forest(
        training_features,
        training_target,
        cv,
    )
    fixed_estimator = tuned_search.best_estimator_
    records: list[dict[str, float | int]] = []
    prediction_records: list[dict[str, object]] = []

    for random_state in range(FIRST_RANDOM_STATE, LAST_RANDOM_STATE + 1):
        estimator = clone(fixed_estimator).set_params(
            random_state=random_state,
            n_jobs=1,
        )
        fold_scores = cross_validate(
            estimator,
            training_features,
            training_target,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
            },
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        accuracy_scores = fold_scores["test_accuracy"]
        macro_f1_scores = fold_scores["test_macro_f1"]

        estimator.fit(training_features, training_target)
        holdout_prediction = estimator.predict(holdout_features)
        holdout_probabilities = estimator.predict_proba(holdout_features)
        class_positions = {
            label: position
            for position, label in enumerate(estimator.classes_)
        }
        record: dict[str, float | int] = {
            "random_state": random_state,
            "configured_random_state_at_run": RANDOM_STATE,
            "cv_accuracy_mean": float(np.mean(accuracy_scores)),
            "cv_accuracy_std": float(np.std(accuracy_scores, ddof=0)),
            "cv_macro_f1_mean": float(np.mean(macro_f1_scores)),
            "cv_macro_f1_std": float(np.std(macro_f1_scores, ddof=0)),
            "holdout_accuracy": float(
                accuracy_score(holdout_target, holdout_prediction)
            ),
            "holdout_balanced_accuracy": float(
                balanced_accuracy_score(holdout_target, holdout_prediction)
            ),
            "holdout_macro_f1": float(
                f1_score(
                    holdout_target,
                    holdout_prediction,
                    labels=list(DECISION_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        for fold_number in range(1, CV_SPLITS + 1):
            score_index = fold_number - 1
            record[f"fold_{fold_number}_accuracy"] = float(
                accuracy_scores[score_index]
            )
            record[f"fold_{fold_number}_macro_f1"] = float(
                macro_f1_scores[score_index]
            )
        records.append(record)

        for row_number, (_, holdout_row) in enumerate(holdout.iterrows()):
            prediction_record: dict[str, object] = {
                "random_state": random_state,
                "configured_random_state_at_run": RANDOM_STATE,
                "meeting_date": holdout_row["meeting_date"],
                "actual_decision": holdout_target.iloc[row_number],
                "predicted_decision": holdout_prediction[row_number],
                "correct": bool(
                    holdout_prediction[row_number] == holdout_target.iloc[row_number]
                ),
            }
            for decision in DECISION_CLASSES:
                prediction_record[f"probability_{decision}"] = float(
                    holdout_probabilities[row_number, class_positions[decision]]
                )
            prediction_records.append(prediction_record)

        if random_state % 10 == 0 or random_state == LAST_RANDOM_STATE:
            print(
                f"Evaluated random_state={random_state}: "
                f"mean_cv_accuracy={record['cv_accuracy_mean']:.4f}, "
                f"mean_cv_macro_f1={record['cv_macro_f1_mean']:.4f}, "
                f"holdout_accuracy={record['holdout_accuracy']:.4f}",
                flush=True,
            )

    results = pd.DataFrame.from_records(records).sort_values("random_state")
    configured_row = results.loc[results["random_state"].eq(RANDOM_STATE)]
    if configured_row.empty:
        raise ValueError(
            f"Configured RANDOM_STATE={RANDOM_STATE} is outside "
            f"{FIRST_RANDOM_STATE}..{LAST_RANDOM_STATE}"
        )
    configured_metrics = {
        metric: float(configured_row[metric].iloc[0])
        for metric in (
            "cv_accuracy_mean",
            "cv_macro_f1_mean",
            "holdout_accuracy",
            "holdout_balanced_accuracy",
            "holdout_macro_f1",
        )
    }
    for metric, configured_value in configured_metrics.items():
        results[f"{metric}_delta_vs_configured_seed"] = (
            results[metric] - configured_value
        )

    context: dict[str, object] = {
        "configured_random_state": RANDOM_STATE,
        "configured_seed_metrics": configured_metrics,
        "best_parameters_held_fixed": tuned_search.best_params_,
        "parameter_search_cv_macro_f1": float(tuned_search.best_score_),
        "cv_splits": CV_SPLITS,
        "training_rows": len(train),
        "holdout_rows": len(holdout),
        "training_first_meeting": train["meeting_date"].min().date().isoformat(),
        "training_last_meeting": train["meeting_date"].max().date().isoformat(),
        "holdout_first_meeting": holdout["meeting_date"].min().date().isoformat(),
        "holdout_last_meeting": holdout["meeting_date"].max().date().isoformat(),
    }
    predictions = pd.DataFrame.from_records(prediction_records)
    return results.reset_index(drop=True), predictions, context


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Atomically write one experiment artifact."""
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
    return path


def save_results(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[Path, Path]:
    """Validate and save aggregate and meeting-level seed results."""
    expected_states = set(range(FIRST_RANDOM_STATE, LAST_RANDOM_STATE + 1))
    if set(results["random_state"]) != expected_states:
        raise ValueError("Results do not contain every random state from 0 to 100")
    if results.isna().any().any():
        raise ValueError("Random-state results contain missing values")
    expected_prediction_rows = len(expected_states) * results.attrs.get(
        "holdout_rows", 0
    )
    if expected_prediction_rows and len(predictions) != expected_prediction_rows:
        raise ValueError("Holdout prediction table has an unexpected row count")
    if predictions.isna().any().any():
        raise ValueError("Holdout prediction table contains missing values")
    if set(predictions["random_state"]) != expected_states:
        raise ValueError("Holdout predictions do not contain every random state")

    results_path = _atomic_write_csv(results, RESULTS_PATH)
    predictions_path = _atomic_write_csv(
        predictions,
        HOLDOUT_PREDICTIONS_PATH,
    )
    return results_path, predictions_path


def main() -> None:
    """Run the fixed 0-through-100 seed sensitivity experiment."""
    results, predictions, context = evaluate_random_states()
    results.attrs["holdout_rows"] = context["holdout_rows"]
    results_path, predictions_path = save_results(results, predictions)

    print("\nRandom-state sensitivity summary")
    print(f"Forward CV splits: {context['cv_splits']}")
    print(f"Training rows: {context['training_rows']}")
    print(f"Diagnostic holdout rows: {context['holdout_rows']}")
    print(f"Fixed forest parameters: {context['best_parameters_held_fixed']}")
    configured_metrics = context["configured_seed_metrics"]
    if not isinstance(configured_metrics, dict):
        raise TypeError("configured_seed_metrics must be a dictionary")
    print(f"Configured seed {context['configured_random_state']}:")
    for metric, label in (
        ("cv_accuracy_mean", "accuracy"),
        ("cv_macro_f1_mean", "macro F1"),
        ("holdout_accuracy", "holdout accuracy (diagnostic)"),
        ("holdout_balanced_accuracy", "holdout balanced accuracy (diagnostic)"),
        ("holdout_macro_f1", "holdout macro F1 (diagnostic)"),
    ):
        seed_values = results[metric]
        print(f"  {label}: {float(configured_metrics[metric]):.4f}")
        print(
            f"  across-seed {label}: mean={seed_values.mean():.4f}, "
            f"std={seed_values.std(ddof=0):.4f}, "
            f"min={seed_values.min():.4f}, max={seed_values.max():.4f}"
        )
    print(f"Saved 101 aggregate seed results to {results_path}")
    print(f"Saved every holdout prediction to {predictions_path}")
    print(
        "Holdout results are diagnostic only. Selecting the best seed from these "
        "columns would turn the holdout into tuning data."
    )


if __name__ == "__main__":
    main()
