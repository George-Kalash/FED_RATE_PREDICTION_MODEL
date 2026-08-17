"""Build and execute a notebook that visualizes holdout model predictions."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

from config import (
    CLEAN_PANEL_PATH,
    DECISION_CLASSES,
    MODEL_PREDICTIONS_PATH,
    PREDICTION_VISUALIZER_PATH,
    PROJECT_DIR,
)


REPOSITORY_DIR = PROJECT_DIR.parent


def load_visualizer_data() -> pd.DataFrame:
    """Load, merge, and validate the holdout predictions and meeting data."""
    missing_paths = [
        path
        for path in (MODEL_PREDICTIONS_PATH, CLEAN_PANEL_PATH)
        if not path.is_file()
    ]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing visualizer inputs: {missing_text}. Run pipeline.py first, "
            "or run clean.py, features.py, and model.py in order."
        )

    predictions = pd.read_csv(
        MODEL_PREDICTIONS_PATH,
        parse_dates=["meeting_date"],
    )
    clean_panel = pd.read_csv(
        CLEAN_PANEL_PATH,
        parse_dates=["meeting_date"],
    )
    prediction_columns = {
        "meeting_date",
        "actual_decision",
        "predicted_decision",
        "probability_cut",
        "probability_hold",
        "probability_hike",
    }
    clean_columns = {
        "meeting_date",
        "policy_rate_after",
        "unemployment",
        "natural_unemployment",
    }
    missing_prediction_columns = sorted(
        prediction_columns - set(predictions.columns)
    )
    missing_clean_columns = sorted(clean_columns - set(clean_panel.columns))
    if missing_prediction_columns or missing_clean_columns:
        raise ValueError(
            "Visualizer input schema mismatch; "
            f"predictions missing={missing_prediction_columns}, "
            f"clean panel missing={missing_clean_columns}"
        )
    if predictions.empty:
        raise ValueError("Prediction table contains no holdout rows")
    if predictions["meeting_date"].duplicated().any():
        raise ValueError("Prediction table contains duplicate meeting dates")

    meeting_values = clean_panel.loc[:, sorted(clean_columns)].copy()
    visualizer_data = predictions.merge(
        meeting_values,
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
        raise ValueError("Clean meeting data is missing values for holdout predictions")
    for decision_column in ("actual_decision", "predicted_decision"):
        if not visualizer_data[decision_column].isin(DECISION_CLASSES).all():
            raise ValueError(f"{decision_column} contains an invalid decision")
    probabilities = visualizer_data[
        ["probability_cut", "probability_hold", "probability_hike"]
    ]
    if not probabilities.apply(lambda column: column.between(0, 1)).all().all():
        raise ValueError("Prediction probabilities must be between zero and one")
    if not (probabilities.sum(axis=1) - 1.0).abs().le(1e-10).all():
        raise ValueError("Prediction probabilities do not sum to one")

    return visualizer_data.sort_values("meeting_date").reset_index(drop=True)


def create_notebook(visualizer_data: pd.DataFrame) -> nbformat.NotebookNode:
    """Create the reader-facing prediction visualization notebook."""
    first_date = visualizer_data["meeting_date"].min().strftime("%B %d, %Y")
    last_date = visualizer_data["meeting_date"].max().strftime("%B %d, %Y")
    correct = int(
        visualizer_data["actual_decision"].eq(
            visualizer_data["predicted_decision"]
        ).sum()
    )
    accuracy = correct / len(visualizer_data)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (project environment)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# FOMC prediction visualizer\n\n"
            "## tl;dr\n\n"
            f"This notebook visualizes **{len(visualizer_data)} chronological "
            f"holdout predictions** from {first_date} through {last_date}. The "
            f"three-class policy matched **{correct} decisions ({accuracy:.1%})**. "
            "Prediction markers are placed on the observed policy-rate and "
            "unemployment series; they do not represent predicted numeric rate or "
            "unemployment levels."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The visualization merges `outputs/predictions.csv` with meeting-level "
            "values from `data/clean/clean_panel.csv`. Only the chronological holdout "
            "is shown. Marker color and shape identify the model's predicted class; "
            "a dark open ring identifies an incorrect class prediction.\n\n"
            "### Key Assumptions\n\n"
            "- The plotted policy rate is the official target midpoint after each meeting.\n"
            "- Unemployment values are the aligned reference-period values in the clean panel.\n"
            "- These are historical holdout predictions, not live next-meeting forecasts.\n"
            "- The confusion matrix contains counts, not normalized percentages."
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
            "from config import CLEAN_PANEL_PATH, DECISION_CLASSES, MODEL_PREDICTIONS_PATH\n\n"
            "predictions = pd.read_csv(MODEL_PREDICTIONS_PATH, parse_dates=['meeting_date'])\n"
            "clean_panel = pd.read_csv(CLEAN_PANEL_PATH, parse_dates=['meeting_date'])\n"
            "meeting_columns = [\n"
            "    'meeting_date', 'policy_rate_after', 'unemployment',\n"
            "    'natural_unemployment'\n"
            "]\n"
            "plot_data = predictions.merge(\n"
            "    clean_panel[meeting_columns], on='meeting_date', how='left',\n"
            "    validate='one_to_one'\n"
            ").sort_values('meeting_date').reset_index(drop=True)\n"
            "probability_columns = ['probability_cut','probability_hold','probability_hike']\n"
            "assert len(plot_data) == len(predictions) and plot_data.meeting_date.is_unique\n"
            "assert plot_data[['policy_rate_after','unemployment','natural_unemployment']].notna().all().all()\n"
            "assert plot_data.actual_decision.isin(DECISION_CLASSES).all()\n"
            "assert plot_data.predicted_decision.isin(DECISION_CLASSES).all()\n"
            "assert np.allclose(plot_data[probability_columns].sum(axis=1), 1.0)\n"
            "plot_data[['meeting_date','actual_decision','predicted_decision',"
            "'policy_rate_after','unemployment']].head()"
        ),
        nbformat.v4.new_markdown_cell(
            "## Results\n\n"
            "### 2. Interest-rate path and predicted decisions\n\n"
            "The line shows the observed post-meeting target midpoint. Prediction "
            "markers show the class selected by the model at each meeting."
        ),
        nbformat.v4.new_code_cell(
            "INK = '#263238'\n"
            "BLUE = '#376B8C'\n"
            "GOLD = '#B28A1E'\n"
            "ORANGE = '#C15C2B'\n"
            "GRID = '#D9DEE3'\n"
            "MUTED = '#73808C'\n"
            "DECISION_STYLE = {\n"
            "    'cut': {'color': ORANGE, 'marker': 'v', 'label': 'Predicted cut'},\n"
            "    'hold': {'color': BLUE, 'marker': 'o', 'label': 'Predicted hold'},\n"
            "    'hike': {'color': GOLD, 'marker': '^', 'label': 'Predicted hike'},\n"
            "}\n\n"
            "fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)\n"
            "ax.step(\n"
            "    plot_data.meeting_date, plot_data.policy_rate_after, where='post',\n"
            "    color=INK, linewidth=2.25, label='Actual target midpoint after meeting'\n"
            ")\n"
            "for decision, style in DECISION_STYLE.items():\n"
            "    selected = plot_data.predicted_decision.eq(decision)\n"
            "    ax.scatter(\n"
            "        plot_data.loc[selected, 'meeting_date'],\n"
            "        plot_data.loc[selected, 'policy_rate_after'],\n"
            "        s=72, marker=style['marker'], color=style['color'],\n"
            "        edgecolor='white', linewidth=0.8, zorder=3, label=style['label']\n"
            "    )\n"
            "incorrect = plot_data.actual_decision.ne(plot_data.predicted_decision)\n"
            "ax.scatter(\n"
            "    plot_data.loc[incorrect, 'meeting_date'],\n"
            "    plot_data.loc[incorrect, 'policy_rate_after'],\n"
            "    s=145, facecolors='none', edgecolors=INK, linewidth=1.4,\n"
            "    zorder=4, label='Incorrect class prediction'\n"
            ")\n"
            "ax.set_title('Federal funds target midpoint and predicted FOMC decisions', loc='left', fontsize=14, weight='bold')\n"
            "ax.set_ylabel('Target midpoint (%)')\n"
            "ax.set_xlabel('FOMC decision date')\n"
            "ax.grid(axis='y', color=GRID, linewidth=0.8)\n"
            "ax.spines[['top','right']].set_visible(False)\n"
            "ax.legend(frameon=False, ncols=2, loc='upper left')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 3. Unemployment path and predicted decisions\n\n"
            "The solid line is aligned unemployment and the dashed reference is the "
            "CBO natural-rate estimate. Prediction markers use the same encoding as "
            "the interest-rate chart."
        ),
        nbformat.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)\n"
            "ax.plot(\n"
            "    plot_data.meeting_date, plot_data.unemployment, color=BLUE,\n"
            "    linewidth=2.25, label='Unemployment rate'\n"
            ")\n"
            "ax.plot(\n"
            "    plot_data.meeting_date, plot_data.natural_unemployment, color=MUTED,\n"
            "    linewidth=1.6, linestyle='--', label='CBO natural unemployment rate'\n"
            ")\n"
            "for decision, style in DECISION_STYLE.items():\n"
            "    selected = plot_data.predicted_decision.eq(decision)\n"
            "    ax.scatter(\n"
            "        plot_data.loc[selected, 'meeting_date'],\n"
            "        plot_data.loc[selected, 'unemployment'],\n"
            "        s=72, marker=style['marker'], color=style['color'],\n"
            "        edgecolor='white', linewidth=0.8, zorder=3, label=style['label']\n"
            "    )\n"
            "ax.scatter(\n"
            "    plot_data.loc[incorrect, 'meeting_date'],\n"
            "    plot_data.loc[incorrect, 'unemployment'],\n"
            "    s=145, facecolors='none', edgecolors=INK, linewidth=1.4,\n"
            "    zorder=4, label='Incorrect class prediction'\n"
            ")\n"
            "ax.set_title('Unemployment and predicted FOMC decisions', loc='left', fontsize=14, weight='bold')\n"
            "ax.set_ylabel('Unemployment rate (%)')\n"
            "ax.set_xlabel('FOMC decision date')\n"
            "ax.grid(axis='y', color=GRID, linewidth=0.8)\n"
            "ax.spines[['top','right']].set_visible(False)\n"
            "ax.legend(frameon=False, ncols=2, loc='upper right')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 4. Three-class confusion matrix\n\n"
            "Rows are actual decisions and columns are model predictions. Every "
            "square contains the number of holdout meetings in that combination; "
            "darker blue means more meetings."
        ),
        nbformat.v4.new_code_cell(
            "decision_order = ['cut', 'hold', 'hike']\n"
            "confusion = pd.crosstab(\n"
            "    plot_data.actual_decision, plot_data.predicted_decision\n"
            ").reindex(index=decision_order, columns=decision_order, fill_value=0)\n"
            "matrix = confusion.to_numpy(dtype=int)\n"
            "fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)\n"
            "image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(1, matrix.max()))\n"
            "colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)\n"
            "colorbar.set_label('Number of meetings')\n"
            "ax.set_xticks(range(3), [label.title() for label in decision_order])\n"
            "ax.set_yticks(range(3), [label.title() for label in decision_order])\n"
            "ax.set_xlabel('Predicted decision')\n"
            "ax.set_ylabel('Actual decision')\n"
            "ax.set_title('FOMC decision confusion matrix (counts)', loc='left', fontsize=14, weight='bold')\n"
            "threshold = matrix.max() / 2 if matrix.max() else 0\n"
            "for row in range(3):\n"
            "    for column in range(3):\n"
            "        value = matrix[row, column]\n"
            "        ax.text(\n"
            "            column, row, str(value), ha='center', va='center',\n"
            "            fontsize=16, weight='bold',\n"
            "            color='white' if value > threshold else INK\n"
            "        )\n"
            "ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)\n"
            "ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)\n"
            "ax.grid(which='minor', color='white', linewidth=2)\n"
            "ax.tick_params(which='minor', bottom=False, left=False)\n"
            "ax.set_aspect('equal')\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell("### 5. Validate plotted counts"),
        nbformat.v4.new_code_cell(
            "assert confusion.shape == (3, 3)\n"
            "assert int(confusion.to_numpy().sum()) == len(plot_data)\n"
            "correct_from_matrix = int(np.trace(confusion.to_numpy()))\n"
            "correct_from_rows = int(plot_data.actual_decision.eq(plot_data.predicted_decision).sum())\n"
            "assert correct_from_matrix == correct_from_rows\n"
            "class_recall = pd.Series({\n"
            "    decision: confusion.loc[decision, decision] / confusion.loc[decision].sum()\n"
            "    for decision in decision_order\n"
            "}, name='recall')\n"
            "pd.DataFrame({\n"
            "    'actual_meetings': confusion.sum(axis=1),\n"
            "    'correct_predictions': pd.Series(np.diag(confusion), index=decision_order),\n"
            "    'recall': class_recall,\n"
            "})"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Read prediction markers as class outcomes, not forecasts of the "
            "numeric policy-rate or unemployment level.\n"
            "- Use the dark error rings to locate cycle points where the model's "
            "predicted class disagreed with the actual decision.\n"
            "- Use the confusion matrix for exact class counts and the validation "
            "table for class recall.\n"
            "- The notebook describes a historical holdout and does not establish "
            "future forecasting accuracy."
        ),
    ]
    return notebook


def execute_and_save_notebook(notebook: nbformat.NotebookNode) -> Path:
    """Execute every cell with the project interpreter and save atomically."""
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
    if chart_outputs < 3:
        raise RuntimeError(
            f"Expected at least three rendered charts, found {chart_outputs}"
        )

    PREDICTION_VISUALIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{PREDICTION_VISUALIZER_PATH.name}.",
            suffix=".tmp",
            dir=PREDICTION_VISUALIZER_PATH.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        nbformat.write(executed, temporary_path)
        temporary_path.replace(PREDICTION_VISUALIZER_PATH)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return PREDICTION_VISUALIZER_PATH


def build_prediction_visualizer() -> Path:
    """Load current artifacts, build the notebook, and execute it top-to-bottom."""
    visualizer_data = load_visualizer_data()
    notebook = create_notebook(visualizer_data)
    return execute_and_save_notebook(notebook)


def main() -> None:
    """Generate the configured prediction visualizer without arguments."""
    output_path = build_prediction_visualizer()
    print(f"Saved executed prediction visualizer to {output_path}")
    print("Rendered charts: interest rate, unemployment, and 3x3 confusion matrix")


if __name__ == "__main__":
    main()
