"""Build and execute a notebook for random-forest holdout results."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from config import (
    CLEAN_PANEL_PATH,
    DECISION_CLASSES,
    FEATURE_COLUMNS,
    PROJECT_DIR,
    RANDOM_FOREST_VISUALIZER_PATH,
    TREE_MODEL_IMPORTANCE_PATH,
    TREE_MODEL_PREDICTIONS_PATH,
)


REPOSITORY_DIR = PROJECT_DIR.parent
TOP_FEATURE_COUNT = 15


def load_random_forest_visualizer_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate random-forest predictions and feature importances."""
    required_paths = (
        TREE_MODEL_PREDICTIONS_PATH,
        TREE_MODEL_IMPORTANCE_PATH,
        CLEAN_PANEL_PATH,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing random-forest visualizer inputs: {missing_text}. "
            "Run features.py and tree_model.py first."
        )

    predictions = pd.read_csv(
        TREE_MODEL_PREDICTIONS_PATH,
        parse_dates=["meeting_date"],
    )
    clean_panel = pd.read_csv(CLEAN_PANEL_PATH, parse_dates=["meeting_date"])
    importance = pd.read_csv(TREE_MODEL_IMPORTANCE_PATH)

    prediction_columns = [
        "meeting_date",
        "actual_decision",
        "random_forest_prediction",
        "random_forest_probability_cut",
        "random_forest_probability_hold",
        "random_forest_probability_hike",
    ]
    clean_columns = [
        "meeting_date",
        "policy_rate_after",
        "unemployment",
        "natural_unemployment",
    ]
    importance_columns = ["feature", "random_forest_importance"]
    missing_prediction_columns = sorted(
        set(prediction_columns) - set(predictions.columns)
    )
    missing_clean_columns = sorted(set(clean_columns) - set(clean_panel.columns))
    missing_importance_columns = sorted(
        set(importance_columns) - set(importance.columns)
    )
    if (
        missing_prediction_columns
        or missing_clean_columns
        or missing_importance_columns
    ):
        raise ValueError(
            "Random-forest visualizer schema mismatch; "
            f"predictions missing={missing_prediction_columns}, "
            f"clean panel missing={missing_clean_columns}, "
            f"importance missing={missing_importance_columns}"
        )
    if predictions.empty:
        raise ValueError("Random-forest prediction table contains no rows")

    prediction_data = predictions.loc[:, prediction_columns].rename(
        columns={
            "random_forest_prediction": "predicted_decision",
            "random_forest_probability_cut": "probability_cut",
            "random_forest_probability_hold": "probability_hold",
            "random_forest_probability_hike": "probability_hike",
        }
    )
    if prediction_data["meeting_date"].isna().any():
        raise ValueError("Random-forest predictions contain invalid meeting dates")
    if prediction_data["meeting_date"].duplicated().any():
        raise ValueError("Random-forest predictions contain duplicate meeting dates")

    visualizer_data = prediction_data.merge(
        clean_panel.loc[:, clean_columns],
        on="meeting_date",
        how="left",
        validate="one_to_one",
    )
    required_values = [
        "policy_rate_after",
        "unemployment",
        "natural_unemployment",
    ]
    if visualizer_data[required_values].isna().any().any():
        raise ValueError("Clean meeting values are missing for forest predictions")
    for decision_column in ("actual_decision", "predicted_decision"):
        if not visualizer_data[decision_column].isin(DECISION_CLASSES).all():
            raise ValueError(f"{decision_column} contains an invalid decision")
    probability_columns = [
        "probability_cut",
        "probability_hold",
        "probability_hike",
    ]
    probabilities = visualizer_data.loc[:, probability_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    if probabilities.isna().any().any():
        raise ValueError("Random-forest probabilities contain invalid values")
    if not probabilities.apply(lambda column: column.between(0, 1)).all().all():
        raise ValueError("Random-forest probabilities must be between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Random-forest probabilities do not sum to one")
    visualizer_data.loc[:, probability_columns] = probabilities

    importance = importance.loc[:, importance_columns].copy()
    importance["feature"] = importance["feature"].astype("string")
    importance["random_forest_importance"] = pd.to_numeric(
        importance["random_forest_importance"], errors="coerce"
    )
    if importance["feature"].isna().any() or importance["feature"].duplicated().any():
        raise ValueError("Random-forest importance features are missing or duplicated")
    expected_features = set(FEATURE_COLUMNS)
    actual_features = set(importance["feature"])
    if actual_features != expected_features:
        raise ValueError(
            "Random-forest importance does not match current FEATURE_COLUMNS; "
            f"missing={sorted(expected_features - actual_features)}, "
            f"unexpected={sorted(actual_features - expected_features)}. "
            "Run tree_model.py again before building the visualizer."
        )
    values = importance["random_forest_importance"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Random-forest importances must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("Random-forest feature importances do not sum to one")
    importance = importance.sort_values(
        ["random_forest_importance", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    return (
        visualizer_data.sort_values("meeting_date").reset_index(drop=True),
        importance,
    )


def create_notebook(
    visualizer_data: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> nbformat.NotebookNode:
    """Create the reader-facing random-forest visualization notebook."""
    first_date = visualizer_data["meeting_date"].min().strftime("%B %d, %Y")
    last_date = visualizer_data["meeting_date"].max().strftime("%B %d, %Y")
    correct = int(
        visualizer_data["actual_decision"].eq(
            visualizer_data["predicted_decision"]
        ).sum()
    )
    accuracy = correct / len(visualizer_data)
    top_features = feature_importance.head(3)["feature"].tolist()
    top_features_text = ", ".join(f"`{feature}`" for feature in top_features)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (project environment)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Random-forest FOMC prediction visualizer\n\n"
            "## tl;dr\n\n"
            f"This notebook visualizes **{len(visualizer_data)} chronological "
            f"random-forest holdout predictions** from {first_date} through "
            f"{last_date}. The predicted class matched **{correct} decisions "
            f"({accuracy:.1%})**. The three highest impurity-based feature "
            f"importances are {top_features_text}."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The notebook uses only the `random_forest_*` fields in "
            "`outputs/tree_model_predictions.csv`, merges meeting values from "
            "`data/clean/clean_panel.csv`, and reads random-forest importance from "
            "`outputs/tree_model_feature_importance.csv`.\n\n"
            "### Key Assumptions\n\n"
            "- The policy-rate line is the observed target midpoint after each meeting.\n"
            "- Unemployment is the aligned value stored in the clean meeting panel.\n"
            "- Markers indicate predicted classes, not predicted numeric rates.\n"
            "- Feature importance is mean decrease in impurity, not a causal effect.\n"
            "- The confusion matrix contains meeting counts, not percentages."
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Load and validate inputs"),
        nbformat.v4.new_code_cell(
            "%matplotlib inline\n"
            "%config InlineBackend.figure_formats = ['svg']\n"
            "from pathlib import Path\n"
            "import sys\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "workspace = Path.cwd().resolve()\n"
            "contents = workspace / 'contents' if (workspace / 'contents').is_dir() else workspace\n"
            "if str(contents) not in sys.path:\n"
            "    sys.path.insert(0, str(contents))\n"
            "from build_random_forest_visualizer import load_random_forest_visualizer_data\n\n"
            "plot_data, feature_importance = load_random_forest_visualizer_data()\n"
            "probability_columns = ['probability_cut', 'probability_hold', 'probability_hike']\n"
            "assert plot_data.meeting_date.is_unique\n"
            "assert np.allclose(plot_data[probability_columns].sum(axis=1), 1.0)\n"
            "assert feature_importance.feature.is_unique\n"
            "assert np.isclose(feature_importance.random_forest_importance.sum(), 1.0)\n"
            "plot_data[['meeting_date', 'actual_decision', 'predicted_decision', "
            "'policy_rate_after', 'unemployment']].head()"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n### 2. Interest-rate path and random-forest decisions\n\n"
            "The line is the observed post-meeting target midpoint. Marker shape "
            "and color identify the random forest's predicted decision."
        ),
        nbformat.v4.new_code_cell(
            "INK = '#263238'\n"
            "BLUE = '#376B8C'\n"
            "GOLD = '#D4AF37'\n"
            "ORANGE = '#178908'\n"
            "GREEN = '#178908'\n"
            "RED = '#CC300D'\n"
            "GRID = '#D9DEE3'\n"
            "MUTED = '#73808C'\n"
            "DECISION_STYLE = {\n"
            "    'cut': {'color': GREEN, 'marker': 'v', 'label': 'Predicted cut'},\n"
            "    'hold': {'color': BLUE, 'marker': 'o', 'label': 'Predicted hold'},\n"
            "    'hike': {'color': RED, 'marker': '^', 'label': 'Predicted hike'},\n"
            "}\n"
            "# Draw holds first and cuts last. Closely spaced emergency meetings can\n"
            "# otherwise let a later hold marker hide the marker for a large cut.\n"
            "DECISION_PLOT_ORDER = ('hold', 'hike', 'cut')\n"
            "incorrect = plot_data.actual_decision.ne(plot_data.predicted_decision)\n\n"
            "fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)\n"
            "ax.step(plot_data.meeting_date, plot_data.policy_rate_after, where='post', "
            "color=INK, linewidth=2.25, zorder=1, "
            "label='Actual target midpoint after meeting')\n"
            "# A shared white halo separates every marker from vertical rate steps.\n"
            "for decision in DECISION_PLOT_ORDER:\n"
            "    style = DECISION_STYLE[decision]\n"
            "    selected = plot_data.predicted_decision.eq(decision)\n"
            "    ax.scatter(plot_data.loc[selected, 'meeting_date'], "
            "plot_data.loc[selected, 'policy_rate_after'], s=116, "
            "marker=style['marker'], color='white', edgecolor='white', "
            "linewidth=0, zorder=2)\n"
            "for layer, decision in enumerate(DECISION_PLOT_ORDER, start=3):\n"
            "    style = DECISION_STYLE[decision]\n"
            "    selected = plot_data.predicted_decision.eq(decision)\n"
            "    ax.scatter(plot_data.loc[selected, 'meeting_date'], "
            "plot_data.loc[selected, 'policy_rate_after'], s=72, "
            "marker=style['marker'], color=style['color'], edgecolor='white', "
            "linewidth=0.8, zorder=layer, label=style['label'])\n"
            "ax.scatter(plot_data.loc[incorrect, 'meeting_date'], "
            "plot_data.loc[incorrect, 'policy_rate_after'], s=145, "
            "facecolors='none', edgecolors=INK, linewidth=1.4, zorder=7, "
            "label='Incorrect class prediction')\n"
            "ax.set_title('Federal funds target midpoint and random-forest decisions', "
            "loc='left', fontsize=14, weight='bold')\n"
            "ax.set_ylabel('Target midpoint (%)')\n"
            "ax.set_xlabel('FOMC decision date')\n"
            "ax.grid(axis='y', color=GRID, linewidth=0.8)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "ax.legend(frameon=False, ncols=2, loc='upper left')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 3. Unemployment path and random-forest decisions\n\n"
            "The solid line is aligned unemployment and the dashed line is the "
            "CBO natural-rate estimate."
        ),
        nbformat.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)\n"
            "ax.plot(plot_data.meeting_date, plot_data.unemployment, color=BLUE, "
            "linewidth=2.25, label='Unemployment rate')\n"
            "ax.plot(plot_data.meeting_date, plot_data.natural_unemployment, "
            "color=MUTED, linewidth=1.6, linestyle='--', "
            "label='CBO natural unemployment rate')\n"
            "for decision, style in DECISION_STYLE.items():\n"
            "    selected = plot_data.predicted_decision.eq(decision)\n"
            "    ax.scatter(plot_data.loc[selected, 'meeting_date'], "
            "plot_data.loc[selected, 'unemployment'], s=72, "
            "marker=style['marker'], color=style['color'], edgecolor='white', "
            "linewidth=0.8, zorder=3, label=style['label'])\n"
            "ax.scatter(plot_data.loc[incorrect, 'meeting_date'], "
            "plot_data.loc[incorrect, 'unemployment'], s=145, facecolors='none', "
            "edgecolors=INK, linewidth=1.4, zorder=4, "
            "label='Incorrect class prediction')\n"
            "ax.set_title('Unemployment and random-forest FOMC decisions', "
            "loc='left', fontsize=14, weight='bold')\n"
            "ax.set_ylabel('Unemployment rate (%)')\n"
            "ax.set_xlabel('FOMC decision date')\n"
            "ax.grid(axis='y', color=GRID, linewidth=0.8)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "ax.legend(frameon=False, ncols=2, loc='upper right')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 4. Random-forest confusion matrix\n\n"
            "Rows are actual decisions and columns are random-forest predictions. "
            "Each square is the number of holdout meetings."
        ),
        nbformat.v4.new_code_cell(
            "decision_order = ['cut', 'hold', 'hike']\n"
            "confusion = pd.crosstab(plot_data.actual_decision, "
            "plot_data.predicted_decision).reindex(index=decision_order, "
            "columns=decision_order, fill_value=0)\n"
            "matrix = confusion.to_numpy(dtype=int)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)\n"
            "image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, matrix.max()))\n"
            "colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)\n"
            "colorbar.set_label('Number of meetings')\n"
            "ax.set_xticks(range(3), [label.title() for label in decision_order])\n"
            "ax.set_yticks(range(3), [label.title() for label in decision_order])\n"
            "ax.set_xlabel('Predicted decision')\n"
            "ax.set_ylabel('Actual decision')\n"
            "ax.set_title('Random-forest FOMC confusion matrix (counts)', "
            "loc='left', fontsize=14, weight='bold')\n"
            "threshold = matrix.max() / 2 if matrix.max() else 0\n"
            "for row in range(3):\n"
            "    for column in range(3):\n"
            "        value = matrix[row, column]\n"
            "        ax.text(column, row, str(value), ha='center', va='center', "
            "fontsize=16, weight='bold', color='white' if value > threshold else INK)\n"
            "ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)\n"
            "ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)\n"
            "ax.grid(which='minor', color='white', linewidth=2)\n"
            "ax.tick_params(which='minor', bottom=False, left=False)\n"
            "ax.set_aspect('equal')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 5. Random-forest feature importance\n\n"
            f"The chart shows the top {TOP_FEATURE_COUNT} features by mean decrease "
            "in impurity. The complete ranking remains available in the source CSV."
        ),
        nbformat.v4.new_code_cell(
            f"top_importance = feature_importance.head({TOP_FEATURE_COUNT}).sort_values(" 
            "'random_forest_importance', ascending=True)\n"
            "bar_colors = [GOLD if feature == feature_importance.iloc[0].feature "
            "else BLUE for feature in top_importance.feature]\n"
            "fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)\n"
            "bars = ax.barh(top_importance.feature, "
            "top_importance.random_forest_importance, color=bar_colors, "
            "edgecolor=INK, linewidth=0.5)\n"
            "ax.bar_label(bars, fmt='%.3f', padding=4, color=INK, fontsize=9)\n"
            "ax.set_title('Random-forest feature importance', loc='left', "
            "fontsize=14, weight='bold')\n"
            "ax.set_xlabel('Mean decrease in impurity (share)')\n"
            "ax.set_ylabel('Feature')\n"
            "ax.set_xlim(0, max(top_importance.random_forest_importance) * 1.18)\n"
            "ax.grid(axis='x', color=GRID, linewidth=0.8)\n"
            "ax.set_axisbelow(True)\n"
            "ax.spines[['top', 'right']].set_visible(False)\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell("### 6. Validate plotted results"),
        nbformat.v4.new_code_cell(
            "assert confusion.shape == (3, 3)\n"
            "assert int(confusion.to_numpy().sum()) == len(plot_data)\n"
            "assert int(np.trace(confusion.to_numpy())) == int(" 
            "plot_data.actual_decision.eq(plot_data.predicted_decision).sum())\n"
            "assert len(top_importance) == min(15, len(feature_importance))\n"
            "class_recall = pd.Series({decision: confusion.loc[decision, decision] "
            "/ confusion.loc[decision].sum() for decision in decision_order}, "
            "name='recall')\n"
            "display(pd.DataFrame({'actual_meetings': confusion.sum(axis=1), "
            "'correct_predictions': pd.Series(np.diag(confusion), "
            "index=decision_order), 'recall': class_recall}))\n"
            "feature_importance.head(10)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- The time-series charts show where the random forest's predicted "
            "decision agreed or disagreed with the observed decision.\n"
            "- The confusion matrix gives exact cut, hold, and hike counts.\n"
            "- Feature importance reports how often and how effectively a feature "
            "reduced impurity across the forest; correlated features can exchange "
            "importance.\n"
            "- These are historical holdout results, not live forecasts or causal claims."
        ),
    ]
    return notebook


def execute_and_save_notebook(notebook: nbformat.NotebookNode) -> Path:
    """Execute every cell and atomically save the completed notebook."""
    execution_environment = os.environ.copy()
    interpreter_directory = str(Path(sys.executable).resolve().parent)
    execution_environment["PATH"] = (
        interpreter_directory
        + os.pathsep
        + execution_environment.get("PATH", "")
    )
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPOSITORY_DIR)}},
        allow_errors=False,
        force_raise_errors=True,
    )
    executed = client.execute(env=execution_environment)

    chart_outputs = 0
    for cell_index, cell in enumerate(executed.cells):
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise RuntimeError(f"Notebook code cell {cell_index} did not execute")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"Notebook code cell {cell_index} saved an error")
            output_data = output.get("data", {})
            if "image/svg+xml" in output_data or "image/png" in output_data:
                chart_outputs += 1
    if chart_outputs < 4:
        raise RuntimeError(
            f"Expected four rendered random-forest charts, found {chart_outputs}"
        )

    RANDOM_FOREST_VISUALIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{RANDOM_FOREST_VISUALIZER_PATH.name}.",
            suffix=".tmp",
            dir=RANDOM_FOREST_VISUALIZER_PATH.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        nbformat.write(executed, temporary_path)
        temporary_path.replace(RANDOM_FOREST_VISUALIZER_PATH)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return RANDOM_FOREST_VISUALIZER_PATH


def build_random_forest_visualizer() -> Path:
    """Build and execute the configured random-forest notebook."""
    visualizer_data, feature_importance = load_random_forest_visualizer_data()
    notebook = create_notebook(visualizer_data, feature_importance)
    return execute_and_save_notebook(notebook)


def main() -> None:
    """Generate the random-forest visualizer without arguments."""
    output_path = build_random_forest_visualizer()
    print(f"Saved executed random-forest visualizer to {output_path}")
    print(
        "Rendered charts: interest rate, unemployment, 3x3 confusion matrix, "
        "and feature importance"
    )


if __name__ == "__main__":
    main()
