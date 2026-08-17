"""Build an executed model diagnostic notebook and portable HTML report.

This script is analysis-only. It reads the saved model artifacts and never
changes the acquisition, cleaning, feature, training, or pipeline stages.
"""

from __future__ import annotations

import json
import math

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_PANEL_PATH,
    MODEL_METRICS_PATH,
    MODEL_PREDICTIONS_PATH,
    OUTPUTS,
    PROJECT_DIR as CONTENTS_DIR,
)

REPOSITORY_DIR = CONTENTS_DIR.parent
METRICS_PATH = MODEL_METRICS_PATH
PREDICTIONS_PATH = MODEL_PREDICTIONS_PATH
NOTEBOOK_PATH = OUTPUTS / "model_accuracy_diagnostics.ipynb"
ARTIFACT_PATH = OUTPUTS / "model_accuracy_artifact.json"
REPORT_PATH = OUTPUTS / "model_accuracy_report.html"


def load_and_validate() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load the saved artifacts and reproduce their headline calculations."""
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["meeting_date"])
    features = pd.read_csv(FEATURE_PANEL_PATH, parse_dates=["meeting_date"])

    holdout = metrics["models"]["is_change"]["holdout"]
    assert len(predictions) == holdout["sample_count"]
    assert predictions["meeting_date"].is_monotonic_increasing
    assert predictions["meeting_date"].is_unique
    assert features.tail(len(predictions))["meeting_date"].reset_index(drop=True).equals(
        predictions["meeting_date"].reset_index(drop=True)
    )

    actual = predictions["actual_is_change"]
    predicted = predictions["predicted_is_change"]
    probability = predictions["probability_change"]
    assert math.isclose(accuracy_score(actual, predicted), holdout["accuracy"])
    assert math.isclose(
        balanced_accuracy_score(actual, predicted), holdout["balanced_accuracy"]
    )
    assert math.isclose(
        brier_score_loss(actual, probability),
        holdout["probability"]["brier_score"],
    )
    assert math.isclose(
        log_loss(actual, probability), holdout["probability"]["log_loss"]
    )
    return metrics, predictions, features


def bootstrap_intervals(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, list[float]]:
    """Return stratified bootstrap intervals for the small holdout."""
    actual = predictions["actual_is_change"].to_numpy()
    predicted = predictions["predicted_is_change"].to_numpy()
    probability = predictions["probability_change"].to_numpy()
    train = features.iloc[: -len(predictions)]
    if train.empty:
        raise ValueError("Feature panel does not contain a training partition")
    train_change_rate = float(train["is_change"].mean())
    baseline_probability = np.full(len(actual), train_change_rate)
    hold_indices = np.flatnonzero(actual == 0)
    change_indices = np.flatnonzero(actual == 1)
    rng = np.random.default_rng(42)
    balanced_scores: list[float] = []
    accuracy_gains: list[float] = []
    brier_gains: list[float] = []
    for _ in range(5_000):
        sample = np.concatenate(
            [
                rng.choice(hold_indices, len(hold_indices), replace=True),
                rng.choice(change_indices, len(change_indices), replace=True),
            ]
        )
        balanced_scores.append(
            balanced_accuracy_score(actual[sample], predicted[sample])
        )
        accuracy_gains.append(
            accuracy_score(actual[sample], predicted[sample])
            - accuracy_score(actual[sample], np.zeros(len(sample), dtype=int))
        )
        brier_gains.append(
            brier_score_loss(actual[sample], baseline_probability[sample])
            - brier_score_loss(actual[sample], probability[sample])
        )
    return {
        "balanced_accuracy": np.quantile(
            balanced_scores, [0.025, 0.5, 0.975]
        ).tolist(),
        "accuracy_gain_vs_hold": np.quantile(
            accuracy_gains, [0.025, 0.5, 0.975]
        ).tolist(),
        "brier_gain_vs_prevalence": np.quantile(
            brier_gains, [0.025, 0.5, 0.975]
        ).tolist(),
    }


def run_exploratory_experiment(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate the documented lagged-history hypothesis on the current holdout."""
    experiment = features.copy()
    signed = experiment["decision"].map({"cut": -1, "hold": 0, "hike": 1})
    experiment["prior_decision"] = signed.shift(1).fillna(0)
    experiment["prior_is_change"] = experiment["is_change"].shift(1).fillna(0)
    experiment["prior2_is_change"] = experiment["is_change"].shift(2).fillna(0)
    experiment["prior3_change_count"] = (
        experiment["is_change"].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    )
    experiment["prior3_direction"] = (
        signed.shift(1).rolling(3, min_periods=1).sum().fillna(0)
    )
    meeting_gap = experiment["meeting_date"].diff().dt.days
    experiment["days_since_prior_meeting"] = meeting_gap.fillna(meeting_gap.median())
    last_change = (
        experiment["meeting_date"]
        .where(experiment["is_change"].eq(1))
        .shift(1)
        .ffill()
    )
    experiment["days_since_prior_change"] = (
        (experiment["meeting_date"] - last_change)
        .dt.days.fillna(3650)
        .clip(upper=3650)
    )

    macro = [
        "rate_level", "rate_chg_1m", "rate_chg_3m", "pce_yoy",
        "pce_yoy_chg", "pce_yoy_chg3", "pce_yoy_ma3", "pce_yoy_ma6",
        "unemployment", "unemp_chg", "unemp_chg3", "natural_unemployment",
        "real_rate_proxy", "labour_gap", "abs_inflation_gap",
    ]
    history = [
        "prior_decision", "prior_is_change", "prior2_is_change",
        "prior3_change_count", "prior3_direction", "days_since_prior_meeting",
        "days_since_prior_change",
    ]
    split_index = len(experiment) - len(predictions)
    train = experiment.iloc[:split_index]
    test = experiment.iloc[split_index:]
    if train.empty or test.empty:
        raise ValueError("Exploratory experiment requires non-empty train and test data")

    def observed_class_balanced_accuracy(
        actual: pd.Series | np.ndarray,
        predicted: pd.Series | np.ndarray,
    ) -> float:
        actual_values = np.asarray(actual)
        predicted_values = np.asarray(predicted)
        return float(
            np.mean(
                [
                    np.mean(predicted_values[actual_values == label] == label)
                    for label in np.unique(actual_values)
                ]
            )
        )

    cv_test_size = len(train) // 6
    if cv_test_size < 1:
        raise ValueError("Training data is too short for five-split validation")
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )
    search = GridSearchCV(
        estimator,
        {"model__C": [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 5, 10]},
        scoring=make_scorer(observed_class_balanced_accuracy),
        cv=TimeSeriesSplit(n_splits=5, test_size=cv_test_size),
        error_score="raise",
    )
    columns = macro + history
    search.fit(train[columns], train["is_change"])
    predicted = search.predict(test[columns])
    probability = search.predict_proba(test[columns])[:, 1]
    return {
        "best_parameters": search.best_params_,
        "accuracy": float(accuracy_score(test["is_change"], predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test["is_change"], predicted)
        ),
        "brier_score": float(brier_score_loss(test["is_change"], probability)),
        "log_loss": float(log_loss(test["is_change"], probability)),
    }


def build_notebook() -> None:
    """Create and execute the reproducible diagnostic notebook."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Fed decision model accuracy diagnostics\n\n"
            "This notebook reproduces the saved holdout metrics, examines error "
            "concentration, measures uncertainty, and records an exploratory "
            "meeting-history hypothesis. It does not alter production files."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, math\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.metrics import (accuracy_score, balanced_accuracy_score, "
            "brier_score_loss, confusion_matrix, log_loss)\n\n"
            "root = Path.cwd()\n"
            "with (root / 'contents/outputs/metrics.json').open() as f:\n"
            "    metrics = json.load(f)\n"
            "pred = pd.read_csv(root / 'contents/outputs/predictions.csv', "
            "parse_dates=['meeting_date'])\n"
            "features = pd.read_csv(root / 'contents/data/clean/feature_panel.csv', "
            "parse_dates=['meeting_date'])\n"
            "len(pred), pred.meeting_date.min(), pred.meeting_date.max()"
        ),
        nbformat.v4.new_markdown_cell("## Metric reproduction and baseline"),
        nbformat.v4.new_code_cell(
            "y = pred.actual_is_change\n"
            "yhat = pred.predicted_is_change\n"
            "p = pred.probability_change\n"
            "train = features.iloc[:-len(pred)]\n"
            "train_rate = train.is_change.mean()\n"
            "summary = pd.DataFrame({\n"
            "    'metric': ['accuracy', 'balanced_accuracy', 'brier', 'log_loss'],\n"
            "    'model': [accuracy_score(y,yhat), balanced_accuracy_score(y,yhat), "
            "brier_score_loss(y,p), log_loss(y,p)],\n"
            "    'baseline': [accuracy_score(y,np.zeros(len(y))), 0.5, "
            "brier_score_loss(y,np.full(len(y),train_rate)), "
            "log_loss(y,np.full(len(y),train_rate))]\n"
            "})\n"
            "summary"
        ),
        nbformat.v4.new_markdown_cell("## Error concentration"),
        nbformat.v4.new_code_cell(
            "binary_confusion = pd.crosstab(y, yhat, rownames=['actual'], "
            "colnames=['predicted'])\n"
            "decision_confusion = pd.crosstab(pred.actual_decision, "
            "pred.predicted_decision, rownames=['actual'], colnames=['predicted'])\n"
            "binary_confusion, decision_confusion"
        ),
        nbformat.v4.new_code_cell(
            "pred.loc[pred.actual_is_change.ne(pred.predicted_is_change), "
            "['meeting_date','actual_decision','predicted_is_change',"
            "'probability_change','predicted_decision']]"
        ),
        nbformat.v4.new_markdown_cell("## Holdout uncertainty"),
        nbformat.v4.new_code_cell(
            "rng = np.random.default_rng(42)\n"
            "a=y.to_numpy(); h=yhat.to_numpy(); idx0=np.flatnonzero(a==0); "
            "idx1=np.flatnonzero(a==1)\n"
            "scores=[]; gains=[]\n"
            "for _ in range(5000):\n"
            "    idx=np.r_[rng.choice(idx0,len(idx0),replace=True), "
            "rng.choice(idx1,len(idx1),replace=True)]\n"
            "    scores.append(balanced_accuracy_score(a[idx],h[idx]))\n"
            "    gains.append(accuracy_score(a[idx],h[idx])-"
            "accuracy_score(a[idx],np.zeros(len(idx),dtype=int)))\n"
            "{'balanced_accuracy_95pct': np.quantile(scores,[.025,.975]), "
            "'accuracy_gain_vs_hold_95pct': np.quantile(gains,[.025,.975])}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Exploratory improvement hypothesis\n\n"
            "The following training-fold-tuned experiment adds only strictly lagged "
            "decision-cycle fields (previous decision/change, recent change count "
            "and direction, meeting spacing, and time since the last change) while "
            "removing several near-duplicate macro columns. Because this specification "
            "was proposed after examining the holdout, its newly computed scores are "
            "exploratory and must be "
            "confirmed with nested expanding-window evaluation or a fresh holdout."
        ),
        nbformat.v4.new_code_cell(
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import brier_score_loss, log_loss, make_scorer\n"
            "from sklearn.model_selection import GridSearchCV, TimeSeriesSplit\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "experiment = features.copy()\n"
            "signed = experiment.decision.map({'cut': -1, 'hold': 0, 'hike': 1})\n"
            "experiment['prior_decision'] = signed.shift(1).fillna(0)\n"
            "experiment['prior_is_change'] = experiment.is_change.shift(1).fillna(0)\n"
            "experiment['prior2_is_change'] = experiment.is_change.shift(2).fillna(0)\n"
            "experiment['prior3_change_count'] = (experiment.is_change.shift(1)"
            ".rolling(3, min_periods=1).sum().fillna(0))\n"
            "experiment['prior3_direction'] = (signed.shift(1).rolling(3, min_periods=1)"
            ".sum().fillna(0))\n"
            "experiment['days_since_prior_meeting'] = experiment.meeting_date.diff().dt.days"
            ".fillna(experiment.meeting_date.diff().dt.days.median())\n"
            "last_change = experiment.meeting_date.where(experiment.is_change.eq(1)).shift(1).ffill()\n"
            "experiment['days_since_prior_change'] = ((experiment.meeting_date-last_change)"
            ".dt.days.fillna(3650).clip(upper=3650))\n"
            "macro = ['rate_level','rate_chg_1m','rate_chg_3m','pce_yoy','pce_yoy_chg',"
            "'pce_yoy_chg3','pce_yoy_ma3','pce_yoy_ma6','unemployment','unemp_chg',"
            "'unemp_chg3','natural_unemployment','real_rate_proxy','labour_gap',"
            "'abs_inflation_gap']\n"
            "history = ['prior_decision','prior_is_change','prior2_is_change',"
            "'prior3_change_count','prior3_direction','days_since_prior_meeting',"
            "'days_since_prior_change']\n"
            "def observed_class_balanced_accuracy(actual, predicted):\n"
            "    actual=np.asarray(actual); predicted=np.asarray(predicted)\n"
            "    return np.mean([np.mean(predicted[actual==label]==label) "
            "for label in np.unique(actual)])\n"
            "estimator = Pipeline([('scale', StandardScaler()), "
            "('model', LogisticRegression(max_iter=5000))])\n"
            "split_index = len(experiment) - len(pred)\n"
            "exp_train, exp_test = experiment.iloc[:split_index], experiment.iloc[split_index:]\n"
            "cv_test_size = len(exp_train) // 6\n"
            "search = GridSearchCV(estimator, {'model__C':[.01,.03,.1,.3,.5,1,2,5,10]}, "
            "scoring=make_scorer(observed_class_balanced_accuracy), "
            "cv=TimeSeriesSplit(n_splits=5, test_size=cv_test_size), error_score='raise')\n"
            "search.fit(exp_train[macro+history], exp_train.is_change)\n"
            "exp_hat = search.predict(exp_test[macro+history])\n"
            "exp_prob = search.predict_proba(exp_test[macro+history])[:,1]\n"
            "exploratory_result = {\n"
            "    'best_parameters': search.best_params_,\n"
            "    'accuracy': accuracy_score(exp_test.is_change, exp_hat),\n"
            "    'balanced_accuracy': balanced_accuracy_score(exp_test.is_change, exp_hat),\n"
            "    'brier_score': brier_score_loss(exp_test.is_change, exp_prob),\n"
            "    'log_loss': log_loss(exp_test.is_change, exp_prob),\n"
            "}\n"
            "assert all(0 <= exploratory_result[name] <= 1 for name in "
            "['accuracy','balanced_accuracy','brier_score'])\n"
            "exploratory_result"
        ),
        nbformat.v4.new_markdown_cell(
            "## Chart map and validation notes\n\n"
            "- **Model versus baseline:** grouped bar; accuracy and balanced accuracy; "
            "supports the limited incremental-gain finding.\n"
            "- **Decision-class recall:** single-series bar; cut/hold/hike; supports the "
            "zero-cut-recall finding.\n"
            "- **QA:** saved metrics were independently recomputed; prediction dates "
            "match the newest feature-panel meetings; all probability scores are "
            "finite and within [0, 1]."
        ),
    ]
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_DIR)}},
    )
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)


def build_artifact(
    metrics: dict,
    predictions: pd.DataFrame,
    intervals: dict[str, list[float]],
    exploratory: dict[str, object],
) -> None:
    """Write the canonical portable-report artifact input."""
    binary = metrics["models"]["is_change"]["holdout"]
    decision = metrics["models"]["decision"]["holdout"]
    generated_at = metrics["generated_at_utc"]
    sample_count = int(binary["sample_count"])
    actual = predictions["actual_is_change"].astype(int)
    predicted = predictions["predicted_is_change"].astype(int)
    model_correct = int(actual.eq(predicted).sum())
    baseline_class = int(binary["majority_baseline"]["class"])
    baseline_correct = int(actual.eq(baseline_class).sum())
    false_negatives = int(actual.eq(1).mul(predicted.eq(0)).sum())
    false_positives = int(actual.eq(0).mul(predicted.eq(1)).sum())
    first_meeting = predictions["meeting_date"].min().strftime("%B %d, %Y")
    last_meeting = predictions["meeting_date"].max().strftime("%B %d, %Y")
    accuracy_gain = binary["accuracy"] - binary["majority_baseline"]["accuracy"]
    balanced_gain = (
        binary["balanced_accuracy"]
        - binary["majority_baseline"]["balanced_accuracy"]
    )
    train_change_rate = (
        metrics["models"]["is_change"]["train_class_counts"]["1"]
        / metrics["models"]["is_change"]["train_sample_count"]
    )
    baseline_log_loss = float(
        log_loss(actual, np.full(sample_count, train_change_rate))
    )
    accuracy_value = float(binary["accuracy"])
    z_score = 1.959963984540054
    wilson_denominator = 1 + z_score**2 / sample_count
    wilson_center = (
        accuracy_value + z_score**2 / (2 * sample_count)
    ) / wilson_denominator
    wilson_half_width = (
        z_score
        * math.sqrt(
            accuracy_value * (1 - accuracy_value) / sample_count
            + z_score**2 / (4 * sample_count**2)
        )
        / wilson_denominator
    )
    wilson_low = wilson_center - wilson_half_width
    wilson_high = wilson_center + wilson_half_width

    comparison_rows = [
        {
            "metric": "Accuracy",
            "series": "Model",
            "value": binary["accuracy"],
            "sample_count": binary["sample_count"],
            "correct_decisions": model_correct,
        },
        {
            "metric": "Accuracy",
            "series": "Always hold",
            "value": binary["majority_baseline"]["accuracy"],
            "sample_count": binary["sample_count"],
            "correct_decisions": baseline_correct,
        },
        {
            "metric": "Balanced accuracy",
            "series": "Model",
            "value": binary["balanced_accuracy"],
            "sample_count": binary["sample_count"],
            "correct_decisions": model_correct,
        },
        {
            "metric": "Balanced accuracy",
            "series": "Always hold",
            "value": binary["majority_baseline"]["balanced_accuracy"],
            "sample_count": binary["sample_count"],
            "correct_decisions": baseline_correct,
        },
    ]
    recall_rows = []
    for label, row in zip(
        decision["confusion_matrix"]["labels"],
        decision["confusion_matrix"]["values"],
        strict=True,
    ):
        total = sum(row)
        position = decision["confusion_matrix"]["labels"].index(label)
        recall_rows.append(
            {
                "decision": label.title(),
                "recall": row[position] / total,
                "actual_meetings": total,
                "correct_meetings": row[position],
            }
        )
    recall_by_decision = {
        row["decision"].lower(): row["recall"] for row in recall_rows
    }
    count_by_decision = {
        row["decision"].lower(): row["actual_meetings"] for row in recall_rows
    }

    missed_changes = predictions.loc[
        predictions["actual_is_change"].eq(1)
        & predictions["predicted_is_change"].eq(0),
        [
            "meeting_date",
            "actual_decision",
            "probability_change",
            "predicted_decision",
            "probability_cut",
            "probability_hold",
            "probability_hike",
        ],
    ].copy()
    missed_changes["meeting_date"] = missed_changes["meeting_date"].dt.strftime(
        "%b %d, %Y"
    )

    recommendation_rows = [
        {
            "priority": 1,
            "change": "Point-in-time data and a fixed prediction cutoff",
            "why": "Current rows use pre-meeting reference periods but today's revised values.",
            "validation": "Rebuild with ALFRED vintages known one day before each meeting.",
        },
        {
            "priority": 2,
            "change": "Market-implied expectations",
            "why": (
                f"Macro levels missed {false_negatives} holdout changes and produced "
                f"{false_positives} false change signals."
            ),
            "validation": "Add lagged Fed funds futures probabilities at the same cutoff.",
        },
        {
            "priority": 3,
            "change": "Meeting-cycle history",
            "why": "Strictly lagged prior decisions can represent hiking and cutting cycles.",
            "validation": "Test prior action, streak, meeting gap, and time since last change.",
        },
        {
            "priority": 4,
            "change": "Broader real-time macro and financial features",
            "why": "PCE and unemployment alone omit wages, payrolls, expectations, and conditions.",
            "validation": "Ablate core PCE, payrolls, claims, 2-year yield, curve, and spreads.",
        },
        {
            "priority": 5,
            "change": "Nested walk-forward tuning and calibration",
            "why": (
                f"The {sample_count}-meeting holdout is too small to choose a "
                "threshold or model safely."
            ),
            "validation": "Tune class weights, threshold, C, and calibration inside outer folds.",
        },
    ]

    sources = [
        {
            "id": "saved_metrics",
            "label": "Saved model metrics",
            "path": "contents/outputs/metrics.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Loads the saved chronological holdout metrics used in the report charts.",
                "tables_used": ["contents/outputs/metrics.json"],
                "filters": ["Chronological holdout: newest 20% of meetings"],
                "metric_definitions": [
                    f"Accuracy is correct hold/change predictions divided by {sample_count} holdout meetings.",
                    "Balanced accuracy is the unweighted mean recall for hold and change.",
                ],
                "sql": (
                    "WITH m AS (SELECT * FROM read_json_auto('contents/outputs/metrics.json')) "
                    "SELECT models.is_change.holdout FROM m"
                ),
            },
        },
        {
            "id": "holdout_predictions",
            "label": "Dated holdout predictions",
            "path": "contents/outputs/predictions.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Selects dated false-negative policy changes from the saved holdout predictions.",
                "tables_used": ["contents/outputs/predictions.csv"],
                "filters": [
                    f"Meetings from {first_meeting} through {last_meeting}",
                    "actual_is_change = 1 AND predicted_is_change = 0",
                ],
                "sql": (
                    "SELECT meeting_date, actual_decision, probability_change, "
                    "predicted_decision, probability_cut, probability_hold, probability_hike "
                    "FROM read_csv_auto('contents/outputs/predictions.csv') "
                    "WHERE actual_is_change = 1 AND predicted_is_change = 0 "
                    "ORDER BY meeting_date"
                ),
            },
        },
        {
            "id": "diagnostic_notebook",
            "label": "Executed accuracy diagnostic notebook",
            "path": "contents/outputs/model_accuracy_diagnostics.ipynb",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Materializes the prioritized validation backlog derived in the diagnostic notebook.",
                "tables_used": ["contents/outputs/model_accuracy_diagnostics.ipynb"],
                "sql": (
                    "SELECT * FROM (VALUES "
                    "(1, 'Point-in-time data and a fixed prediction cutoff'), "
                    "(2, 'Market-implied expectations'), "
                    "(3, 'Meeting-cycle history'), "
                    "(4, 'Broader real-time macro and financial features'), "
                    "(5, 'Nested walk-forward tuning and calibration')) "
                    "AS recommendations(priority, change) ORDER BY priority"
                ),
            },
        },
        {
            "id": "alfred_docs",
            "label": "St. Louis Fed ALFRED documentation",
            "href": "https://fred.stlouisfed.org/docs/api/fred/alfred.html",
        },
        {
            "id": "fred_observations_docs",
            "label": "FRED series observations API documentation",
            "href": "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
        },
        {
            "id": "cme_fedwatch",
            "label": "CME FedWatch",
            "href": "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        },
        {
            "id": "nyfed_expectations",
            "label": "New York Fed Survey of Market Expectations",
            "href": "https://www.newyorkfed.org/markets/market-intelligence/survey-of-market-expectations",
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Improving Fed Decision Model Accuracy",
            "description": "Technical diagnosis and prioritized accuracy roadmap.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "model_baseline_chart",
                    "title": "Holdout model and baseline scores",
                    "subtitle": f"Newest {sample_count} meetings; higher is better for both measures.",
                    "type": "bar",
                    "dataset": "model_comparison",
                    "sourceId": "saved_metrics",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "label": "Score",
                            "format": "percent",
                        },
                        "color": {"field": "series", "type": "nominal", "label": "Series"},
                        "tooltip": [
                            {"field": "sample_count", "type": "quantitative", "label": "Meetings"},
                            {"field": "correct_decisions", "type": "quantitative", "label": "Correct"},
                        ],
                    },
                    "options": {"grouping": "grouped", "legend": {"show": True}},
                    "valueFormat": "percent",
                },
                {
                    "id": "decision_recall_chart",
                    "title": "Three-class recall by actual decision",
                    "subtitle": (
                        "Recall is shown separately for cuts, holds, and hikes in the holdout."
                    ),
                    "type": "bar",
                    "dataset": "decision_recall",
                    "sourceId": "saved_metrics",
                    "encodings": {
                        "x": {"field": "decision", "type": "nominal", "label": "Actual decision"},
                        "y": {
                            "field": "recall",
                            "type": "quantitative",
                            "label": "Recall",
                            "format": "percent",
                        },
                        "tooltip": [
                            {"field": "actual_meetings", "type": "quantitative", "label": "Actual meetings"},
                            {"field": "correct_meetings", "type": "quantitative", "label": "Correct"},
                        ],
                    },
                    "valueFormat": "percent",
                },
            ],
            "tables": [
                {
                    "id": "missed_changes_table",
                    "title": "Missed policy changes",
                    "subtitle": (
                        f"{false_negatives} actual changes were predicted as holds in the current holdout."
                    ),
                    "dataset": "missed_changes",
                    "sourceId": "holdout_predictions",
                    "defaultSort": {"field": "meeting_date", "direction": "asc"},
                    "columns": [
                        {"field": "meeting_date", "label": "Meeting", "type": "text"},
                        {"field": "actual_decision", "label": "Actual", "type": "text"},
                        {"field": "predicted_decision", "label": "3-class prediction", "type": "text"},
                        {"field": "probability_change", "label": "P(change)", "format": "percent"},
                        {"field": "probability_cut", "label": "P(cut)", "format": "percent"},
                        {"field": "probability_hold", "label": "P(hold)", "format": "percent"},
                    ],
                },
                {
                    "id": "recommendations_table",
                    "title": "Prioritized validation backlog",
                    "subtitle": "Order reflects expected information gain, not guaranteed accuracy lift.",
                    "dataset": "recommendations",
                    "sourceId": "diagnostic_notebook",
                    "defaultSort": {"field": "priority", "direction": "asc"},
                    "columns": [
                        {"field": "priority", "label": "Priority", "type": "number"},
                        {"field": "change", "label": "Change", "type": "text"},
                        {"field": "why", "label": "Why", "type": "text"},
                        {"field": "validation", "label": "How to validate", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Improving Fed Decision Model Accuracy"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "saved_metrics",
                    "body": (
                        "## Technical summary\n\n"
                        "The current binary model is useful but not yet stable enough to optimize against its headline accuracy. "
                        f"It correctly classified **{model_correct} of {sample_count} meetings ({binary['accuracy']:.1%})**, "
                        f"compared with **{baseline_correct}** for the majority-class rule "
                        f"({binary['majority_baseline']['accuracy']:.1%}). "
                        f"Balanced accuracy is **{binary['balanced_accuracy']:.1%}**, but a stratified bootstrap places its "
                        "approximate 95% interval at "
                        f"**{intervals['balanced_accuracy'][0]:.1%}–{intervals['balanced_accuracy'][2]:.1%}**. "
                        "The immediate objective should be better time-valid information and evaluation, with market expectations and "
                        "meeting-cycle state added before trying a more complex model."
                    ),
                },
                {
                    "id": "baseline_evidence",
                    "type": "markdown",
                    "sourceId": "saved_metrics",
                    "body": (
                        "## The accuracy gain is real-looking but statistically fragile\n\n"
                        f"The model improves balanced accuracy by {balanced_gain:.1%} over the majority baseline, while the raw accuracy "
                        f"gain is {accuracy_gain:.1%}. Its probability ranking has ROC AUC "
                        f"{binary['probability']['roc_auc']:.1%}; its log loss is {binary['probability']['log_loss']:.3f}, compared with "
                        f"{baseline_log_loss:.3f} when predicting the training change rate for every meeting. That combination helps "
                        "separate ranking quality from probability calibration: the model "
                        "often orders meetings sensibly but becomes too confident."
                    ),
                },
                {"id": "baseline_chart", "type": "chart", "chartId": "model_baseline_chart", "layout": "full"},
                {
                    "id": "error_pattern",
                    "type": "markdown",
                    "sourceId": "holdout_predictions",
                    "body": (
                        "## Regime turns—not routine holds—drive the failures\n\n"
                        f"The three-class model recalled {recall_by_decision.get('hike', 0.0):.1%} of hikes, "
                        f"{recall_by_decision.get('hold', 0.0):.1%} of holds, and "
                        f"{recall_by_decision.get('cut', 0.0):.1%} of cuts. The binary model missed "
                        f"{false_negatives} actual changes and produced {false_positives} false change signals. "
                        "These are timing errors around cycle transitions, exactly where "
                        "slow monthly macro levels contain less information than market pricing, recent policy path, and communications."
                    ),
                },
                {"id": "recall_chart", "type": "chart", "chartId": "decision_recall_chart", "layout": "full"},
                {"id": "missed_changes", "type": "table", "tableId": "missed_changes_table", "layout": "full"},
                {
                    "id": "data_priority",
                    "type": "markdown",
                    "body": (
                        "## Improve the information set before the algorithm\n\n"
                        "First rebuild macro inputs with [ALFRED vintages](https://fred.stlouisfed.org/docs/api/fred/alfred.html) known at a "
                        "fixed cutoff such as 4:00 p.m. ET on the day before each meeting. FRED's observations endpoint supports real-time "
                        "periods and vintage dates. Then add pre-cutoff market-implied probabilities from "
                        "[CME FedWatch](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html) or licensed Fed funds futures "
                        "settlements. The [New York Fed expectation surveys](https://www.newyorkfed.org/markets/market-intelligence/"
                        "survey-of-market-expectations) are useful research benchmarks, but publication timing must be checked before "
                        "using a result as a live feature."
                    ),
                },
                {
                    "id": "feature_priority",
                    "type": "markdown",
                    "sourceId": "diagnostic_notebook",
                    "body": (
                        "## Add cycle state and remove redundant representations\n\n"
                        "Add only lagged meeting features: previous action and size, consecutive actions in the same direction, meetings and "
                        "days since the last change, days since the prior meeting, and whether the meeting is scheduled. An exploratory "
                        f"variant combining these fields with a de-duplicated macro set reached **{exploratory['accuracy']:.1%} "
                        f"accuracy**, **{exploratory['balanced_accuracy']:.1%} balanced accuracy**, and a "
                        f"**{exploratory['brier_score']:.3f} Brier score** on this same holdout. This is a hypothesis, not a "
                        "validated gain, because "
                        "the variant was proposed after the holdout errors were inspected."
                    ),
                },
                {
                    "id": "modeling_priority",
                    "type": "markdown",
                    "body": (
                        "## Keep the model small, coherent, and calibrated\n\n"
                        "Use one coherent three-class probability model, then derive hold/change by summing hike and cut probabilities. "
                        "This prevents the current binary and three-class predictions from disagreeing. Compare elastic-net logistic "
                        "regression with one shallow tree-based benchmark, but tune class weights, regularization, thresholds, and "
                        "calibration only inside nested expanding-window folds. Optimize balanced accuracy and macro F1 alongside Brier "
                        "score and log loss; raw accuracy alone rewards predicting hold."
                    ),
                },
                {"id": "recommendations", "type": "table", "tableId": "recommendations_table", "layout": "full"},
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "diagnostic_notebook",
                    "body": (
                        "## Model validation plan\n\n"
                        "1. Freeze a prediction timestamp and construct every feature as it existed then.\n"
                        "2. Use nested expanding-window backtests; keep the newest untouched meetings as a final confirmation set.\n"
                        "3. Compare against always-hold, previous-decision, and market-implied baselines.\n"
                        "4. Report recall for cut, hold, and hike; calibration curves; Brier score; log loss; and confidence intervals.\n"
                        "5. Promote a change only if it improves several outer folds and does not collapse cut recall or calibration."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "saved_metrics",
                    "body": (
                        "## Limitations and robustness\n\n"
                        f"The holdout has only {sample_count} meetings, including {count_by_decision.get('cut', 0)} cuts. The accuracy "
                        f"Wilson interval is roughly {wilson_low:.1%}–{wilson_high:.1%}, and the bootstrap interval for accuracy "
                        f"improvement over the majority baseline is {intervals['accuracy_gain_vs_hold'][0]:.1%}–"
                        f"{intervals['accuracy_gain_vs_hold'][2]:.1%}. Current FRED observations are revised "
                        "rather than point-in-time vintages. The 19 features also contain exact or near-exact representations of the same "
                        "signals—for example PCE inflation and its two-percent gap—which weakens coefficient interpretation and can amplify "
                        "regime-specific confidence."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        "Implement the validation harness first, then add meeting-cycle features as the first controlled experiment. After "
                        "that, add ALFRED vintages and a market-implied expectation feature. Do not choose a new threshold or model from the "
                        f"current {sample_count} meetings; reserve a genuinely untouched period or use nested outer folds for the "
                        "acceptance decision.\n\n"
                        "### Further questions\n\n"
                        "- Is the intended prediction timestamp one day, one week, or one month before the meeting?\n"
                        "- Is the goal best top-class accuracy, reliable probabilities, or early detection of policy turns?\n"
                        "- Can the project license historical Fed funds futures settlements, or must all inputs remain free?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "model_comparison": comparison_rows,
                "decision_recall": recall_rows,
                "missed_changes": missed_changes.to_dict(orient="records"),
                "recommendations": recommendation_rows,
            },
        },
        "sources": sources,
    }
    with ARTIFACT_PATH.open("w", encoding="utf-8") as artifact_file:
        json.dump(artifact, artifact_file, indent=2, allow_nan=False)
        artifact_file.write("\n")


def main() -> None:
    metrics, predictions, features = load_and_validate()
    intervals = bootstrap_intervals(predictions, features)
    exploratory = run_exploratory_experiment(features, predictions)
    build_notebook()
    build_artifact(metrics, predictions, intervals, exploratory)
    print(f"Saved executed notebook to {NOTEBOOK_PATH}")
    print(f"Saved report artifact to {ARTIFACT_PATH}")
    print(f"Portable report target: {REPORT_PATH}")


if __name__ == "__main__":
    main()
