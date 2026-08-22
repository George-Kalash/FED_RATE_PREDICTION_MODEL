# Federal Reserve Decision Predictor

This project builds a chronological machine-learning pipeline that predicts
whether an FOMC meeting results in a **cut**, **hold**, or **hike**. It downloads
Federal Reserve and FRED data, aligns only information available before each
meeting, engineers macroeconomic, Treasury-market, and prior-decision features,
then tunes a random-forest classifier with forward time-series
cross-validation. It is an educational historical backtest, not financial
advice or a live trading system.

## Results

The current feature panel contains 273 FOMC decisions and 38 features. The
oldest 80% is used for training and the newest 20% is kept as a chronological
holdout.

| Model | Training CV macro F1 | Holdout accuracy | Balanced accuracy | Holdout macro F1 |
|---|---:|---:|---:|---:|
| Random forest | 0.797 | 0.927 | 0.964 | 0.919 |

### Random-forest confusion matrix

Rows are actual decisions and columns are predictions.

| Actual \ Predicted | Cut | Hold | Hike |
|---|---:|---:|---:|
| Cut | 7 | 0 | 0 |
| Hold | 1 | 33 | 3 |
| Hike | 0 | 0 | 11 |

These results cover the current 55-meeting holdout from October 2019 through
July 2026. A small historical holdout can change materially after only a few
predictions, and present-day FRED values are not point-in-time economic
vintages.

## How it works

```text
pull_from_apis.py -> clean.py -> features.py -> tree_model.py
      acquire          align       engineer       train/evaluate
```

- Policy labels come from the official federal-funds target before and after
  each FOMC decision.
- Inputs include PCE inflation, unemployment, the CBO natural unemployment
  rate, Treasury yields and spreads, and strictly lagged meeting history.
- Daily market observations must occur before the meeting date.
- The train/test split and all cross-validation folds preserve chronology.
- Hyperparameters are selected on training folds; the final holdout is reported
  only after fitting.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Create `contents/.env` and add your FRED API key:

```dotenv
FRED_API_KEY=your_api_key_here
```

## Run the complete model

```bash
cd contents
python3 pipeline.py
```

No destination arguments are required. The pipeline refreshes raw inputs,
builds the clean and feature panels, tunes the random forest, and writes
the evaluation artifacts.

To rerun only the random forest against an existing feature panel:

```bash
python3 tree_model.py
```

To test random-forest sensitivity across random states 0 through 100:

```bash
python3 test_splits.py
```

To regenerate the random-forest charts notebook:

```bash
python3 build_random_forest_visualizer.py
```

Open
[random_forest_visualizer.ipynb](contents/outputs/random_forest_visualizer.ipynb)
to inspect the policy-rate chart, unemployment chart, 3×3 confusion matrix,
and feature-importance chart.

## Main outputs

| File | Purpose |
|---|---|
| `contents/data/clean/clean_panel.csv` | Meeting-aligned data and actual decisions |
| `contents/data/clean/feature_panel.csv` | Final pre-meeting feature matrix |
| `contents/outputs/tree_model_metrics.json` | CV settings, selected parameters, and holdout metrics |
| `contents/outputs/tree_model_predictions.csv` | Meeting-level random-forest predictions and probabilities |
| `contents/outputs/tree_model_feature_importance.csv` | Random-forest feature importances |
| `contents/outputs/tree_model_factor_rankings.csv` | Most and least influential features |
| `contents/outputs/random_forest_*.png` | Four exported charts from the visualization notebook |
| `contents/outputs/random_state_cv_results.csv` | Per-seed and per-fold random-forest metrics |
| `contents/outputs/random_state_holdout_predictions.csv` | Meeting-level seed-sensitivity predictions |

Generated data and outputs are ignored by Git. To clear them while preserving
the directory structure:

```bash
python3 clean_data_and_outputs.py
```

## Core files

```text
contents/
├── config.py
├── pull_from_apis.py
├── scrape.py
├── clean.py
├── features.py
├── tree_model.py
├── pipeline.py
├── test_splits.py
├── build_random_forest_visualizer.py
└── clean_data_and_outputs.py
```
