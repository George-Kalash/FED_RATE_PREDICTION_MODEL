# Federal Reserve Decision Predictor

An educational, reproducible pipeline for modeling Federal Open Market
Committee (FOMC) policy decisions from U.S. policy-rate, inflation, labour, and
strictly pre-meeting Treasury-market and meeting-history data.

The project follows four modeling stages:

```text
acquire -> clean and align -> engineer features -> train and evaluate
```

It uses a hierarchical logistic-regression model:

1. Estimate whether the FOMC will **hold or change** its target.
2. Conditional on a change, estimate whether the decision is a **cut or hike**.
3. Combine those estimates into coherent cut, hold, and hike probabilities.

The repository also contains an optional supplementary web scraper and an
accuracy-report builder. Scraped website content is documentation-only and is
not used as model input.

> **Important:** this is a research and learning project, not financial advice
> or a production forecasting system. The current training stage writes
> predictions for a historical chronological holdout. It does not serialize a
> deployed estimator or automatically forecast the next FOMC meeting.

## Contents

- [How the pipeline works](#how-the-pipeline-works)
- [Data sources](#data-sources)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the project](#running-the-project)
- [Cleaning and labels](#cleaning-and-labels)
- [Model features](#model-features)
- [Model design](#model-design)
- [Generated artifacts](#generated-artifacts)
- [Accuracy report](#accuracy-report)
- [Prediction visualizer](#prediction-visualizer)
- [Supplementary scraping](#supplementary-scraping)
- [Validation and leakage controls](#validation-and-leakage-controls)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Extending the project](#extending-the-project)

## How the pipeline works

```text
Official structured inputs
  |- FRED policy target series
  |- FRED PCE, unemployment, and natural unemployment
  |- FRED Treasury yields, curve spreads, and research-only market series
  `- Official Federal Reserve FOMC meeting calendar
                |
                v
       data/raw/*.csv
                |
                v
  clean.py: splice policy regimes, calculate PCE inflation,
            align macro data to meetings, construct labels
                |
                v
       data/clean/clean_panel.csv
                |
                v
  features.py: macro momentum, rolling averages, policy context,
               prior-market closes, and lagged meeting-cycle features
                |
                v
       data/clean/feature_panel.csv
                |
                v
  model.py: chronological holdout, forward CV, hierarchical
            logistic regressions, threshold policy, evaluation
                |
                v
       outputs/metrics.json
       outputs/coefficients.csv
       outputs/predictions.csv
                |
                v
  build_accuracy_report.py: executed notebook, report artifact,
                            and self-contained HTML report
```

The optional `scrape.py` path is separate. It produces coverage artifacts but
does not connect to `clean.py`, `features.py`, or `model.py`.

## Data sources

### Structured modeling inputs

The following declarations live in `contents/config.py` and form the complete
structured modeling backbone.

| Logical name | Provider | Series | Frequency | Purpose |
|---|---|---:|---|---|
| `target_rate` | FRED | `DFEDTAR` | Daily | Legacy federal funds point target before the target-range regime |
| `target_lower` | FRED | `DFEDTARL` | Daily | Lower bound of the federal funds target range |
| `target_upper` | FRED | `DFEDTARU` | Daily | Upper bound of the federal funds target range |
| `pce_index` | FRED | `PCEPI` | Monthly | PCE price-index level used to calculate year-over-year inflation |
| `unemployment` | FRED | `UNRATE` | Monthly | U-3 civilian unemployment rate, seasonally adjusted |
| `natural_unemployment` | FRED | `NROU` | Quarterly | CBO estimate of the natural rate of unemployment |
| `treasury_3m_pct` through `treasury_30y_pct` | FRED | `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS5`, `DGS10`, `DGS30` | Daily | Nominal Treasury yield curve |
| `curve_10y_minus_2y_pct`, `curve_10y_minus_3m_pct` | FRED | `T10Y2Y`, `T10Y3M` | Daily | Treasury curve slopes |
| `real_5y_tips_pct`, `real_10y_tips_pct`, `real_30y_tips_pct` | FRED | `DFII5`, `DFII10`, `DFII30` | Daily | Inflation-indexed Treasury yields; acquired for research |
| `breakeven_inflation_5y_pct`, `breakeven_inflation_10y_pct` | FRED | `T5YIE`, `T10YIE` | Daily | Market-implied inflation; acquired for research |
| `investment_grade_oas_pct`, `high_yield_oas_pct` | FRED | `BAMLC0A0CM`, `BAMLH0A0HYM2` | Daily | Corporate credit spreads; acquired for research |
| FOMC calendar | Federal Reserve | Official current and historical calendar pages | Meeting-level | Decision dates, scheduled status, and auditable source URLs |

FRED observations are downloaded from:

```text
https://api.stlouisfed.org/fred/series/observations
```

The FOMC calendar parser reads the current calendar and official historical
pages. For a two-day meeting, the final day is stored as the decision date.
Historical conference calls are retained only when the official page identifies
a policy statement.

All 16 declared market series are downloaded. The estimator currently uses the
six nominal Treasury maturities from 3 months through 10 years and the two
declared curve spreads. This subset preserves meetings before and after 2008.
TIPS and breakeven series begin in 2003, the configured corporate-spread series
currently begin in 2023, and the 30-year Treasury series has a 2002-2006
publication gap, so those series remain raw research inputs rather than forcing
the training panel to discard earlier meetings or silently impute long gaps.

### Supplementary coverage sources

`SUPPLEMENTARY_SOURCES` contains more than 40 Federal Reserve, BLS, BEA,
Treasury, CBO, banking, market, and international-organization pages. The
scraper records page titles, headings, and small table previews.

These records are **not model features**. A scraped source must not be merged
into the feature panel unless it receives:

- a precise numeric feature definition;
- an observation and release timestamp;
- a documented transformation;
- a missing-data policy; and
- a leakage test proving the value was available before the prediction cutoff.

## Project structure

```text
FED_decision_predictor/
|- README.md
|- requirements.txt
|- contents/
|  |- .env                         # local secret; ignored by Git
|  |- .gitignore
|  |- config.py                    # paths, series, sources, features, tuning grids
|  |- pull_from_apis.py            # FRED and FOMC calendar acquisition
|  |- scrape.py                    # optional coverage-only HTML scraping
|  |- clean.py                     # policy splice, macro alignment, labels
|  |- features.py                  # leakage-controlled feature engineering
|  |- model.py                     # hierarchical models and evaluation
|  |- pipeline.py                  # configured end-to-end orchestration
|  |- clean_data_and_outputs.py    # remove generated files, preserve directories
|  |- build_prediction_visualizer.py # executed prediction charts notebook
|  |- build_accuracy_report.py     # notebook, JSON report, and HTML report
|  |- check_links.py               # unfinished supplementary-link QA scaffold
|  |- fed_rate_prediction_resources.csv
|  |- fed_rate_prediction_resources.json
|  |- data/
|  |  |- raw/                      # generated acquisition files; ignored by Git
|  |  `- clean/                    # generated panels; ignored by Git
|  `- outputs/                     # generated model/report artifacts; ignored by Git
`- log.txt                         # local project log, when used
```

The stage scripts use paths declared in `config.py`. The normal pipeline does
not require output-directory command-line arguments.

## Installation

### Prerequisites

- Python 3.11 or newer is recommended.
- A free FRED API key.
- Internet access for FRED and Federal Reserve acquisition.
- Node.js for generating the portable HTML accuracy report.

### Create the environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The requirements include pandas, NumPy, scikit-learn, requests, Beautiful Soup,
lxml, python-dotenv, Matplotlib, nbformat, nbclient, and an IPython kernel.

### Configure the FRED key

Create `contents/.env`:

```dotenv
FRED_API_KEY=your_fred_api_key_here
```

Quotes are optional. `python-dotenv` removes normal dotenv quoting, while the
key resolver also strips accidental surrounding whitespace. Never commit the
real key. `contents/.gitignore` excludes `.env` and `.env.*` files.

## Configuration

Central configuration lives in `contents/config.py`.

### Important pipeline constants

| Constant | Meaning |
|---|---|
| `PIPELINE_START_YEAR` | First FOMC calendar year requested by `pipeline.py` |
| `PIPELINE_END_YEAR` | Last calendar year; defaults to the current year |
| `PIPELINE_FRED_OBSERVATION_START` | Optional FRED observation lower bound; `None` requests all available history |
| `PIPELINE_SCRAPE_COVERAGE` | Whether the end-to-end pipeline also runs the optional coverage scraper |
| `FRED_SERIES` | Logical input names, FRED IDs, frequencies, and modeling roles |
| `BOND_FEATURE_COLUMNS` | Market series permitted to enter the current estimator |
| `BOND_FEATURE_MAX_AGE_DAYS` | Maximum accepted age of a prior-market observation |
| `FEATURE_COLUMNS` | Exact ordered model feature schema |
| `MODEL_TEST_FRACTION` | Newest chronological fraction reserved as holdout |
| `CV_SPLITS` | Number of forward cross-validation splits |
| `LOGISTIC_C_VALUES` | Logistic-regression regularization candidates |

### Decision-policy constants

The configuration also declares candidate class weights and thresholds for:

- change versus hold;
- cut versus hike conditional on a change;
- minimum change probability for the cut override;
- minimum conditional cut probability; and
- minimum joint cut probability.

All candidate selection occurs on training data. Do not manually select a value
after inspecting the final holdout and then report that holdout as untouched.

## Running the project

Run commands from `contents/` unless noted otherwise:

```bash
cd contents
```

### Complete configured pipeline

```bash
python3 pipeline.py
```

This performs:

1. FRED series acquisition;
2. FOMC calendar acquisition;
3. optional coverage scraping if enabled in `config.py`;
4. cleaning and meeting alignment;
5. feature engineering; and
6. model training and evaluation.

The pipeline reads the API key from `contents/.env`, uses configured paths, and
does not require command-line destination arguments.

### Run stages separately

```bash
python3 pull_from_apis.py
python3 clean.py
python3 features.py
python3 model.py
python3 build_prediction_visualizer.py
python3 build_accuracy_report.py
```

Stage dependencies are strict:

| Script | Requires | Main outputs |
|---|---|---|
| `pull_from_apis.py` | FRED key and network | 22 raw FRED CSVs and `fomc_meetings.csv` |
| `clean.py` | Complete raw inputs | `clean_panel.csv` |
| `features.py` | Clean panel plus raw histories | `feature_panel.csv` |
| `model.py` | Feature panel | Metrics, coefficients, predictions |
| `build_prediction_visualizer.py` | Predictions and clean panel | Executed visualization notebook |
| `build_accuracy_report.py` | Model outputs and feature panel | Notebook, report JSON, HTML |

### Structured acquisition options

The normal pipeline uses constants, but `pull_from_apis.py` also exposes a
standalone acquisition CLI:

```bash
python3 pull_from_apis.py \
  --observation-start 1990-01-01 \
  --start-year 2000 \
  --end-year 2026
```

Available options:

- `--api-key`: one-run override; avoid placing secrets in shell history;
- `--observation-start`: optional ISO date in `YYYY-MM-DD` format;
- `--start-year`: first official calendar year to parse; and
- `--end-year`: final calendar year to parse.

Prefer `.env` over `--api-key` for normal use.

### Remove generated data and outputs

To delete acquired data, cleaned panels, model outputs, and generated reports:

```bash
python3 clean_data_and_outputs.py
```

This is destructive. It cleans only the configured `data/raw`, `data/clean`,
and `outputs` directories, refuses symlinked or unapproved targets, and preserves
their `.gitkeep` files. Source code, configuration, `.env`, and documentation
are not removed.

## Cleaning and labels

### Policy target across the 2008 regime change

The official target representation changes over the sample:

- Before the target-range regime, `DFEDTAR` supplies the point target.
- Once both range bounds are available, the policy level becomes:

```text
policy_rate = (target_lower + target_upper) / 2
```

The clean data retains `policy_regime`, lower bound, upper bound, and midpoint
fields so the splice remains auditable. It does not substitute the effective
federal funds rate for the official target.

### Inflation

PCE index observations are normalized to month-end. Year-over-year inflation is:

```text
pce_yoy[t] = (pce_index[t] / pce_index[t-12] - 1) * 100
```

A value is produced only when the complete required monthly history exists.

### Labour data

Monthly unemployment supplies the labour calendar. Quarterly NROU values are
normalized to month-end and carried forward only into later months. NROU is
never backfilled into dates before its first observation, and projections after
the last unemployment observation are excluded.

### Meeting alignment

Each meeting receives independently aligned macro values using the latest
eligible month-end reference period on or before the meeting. The policy target
is aligned strictly before and strictly after the decision date.

The clean panel retains reference dates, allowing validators to reject future
reference periods. However, a FRED observation date is a reference period—not a
release timestamp. See [Known limitations](#known-limitations).

### Decision labels

The label compares the official policy rate immediately after the meeting with
the rate immediately before it:

```text
rate_change_bps = (policy_rate_after - policy_rate_before) * 100
```

Using `RATE_CHANGE_TOLERANCE_BPS`:

- positive change above tolerance -> `hike`;
- negative change below tolerance -> `cut`;
- change within tolerance -> `hold`; and
- `is_change = 1` for cuts or hikes, otherwise `0`.

Only target and audit columns may use the post-meeting policy rate. Post-meeting
fields are excluded from the feature panel.

## Model features

The configured model uses 38 numeric features.

### Policy-rate features

| Feature | Definition |
|---|---|
| `rate_level` | Official target level strictly before the meeting |
| `rate_chg_1m` | Change from the target in force one calendar month earlier |
| `rate_chg_3m` | Change from the target in force three calendar months earlier |

### Inflation features

| Feature | Definition |
|---|---|
| `pce_yoy` | Twelve-month PCE index inflation |
| `pce_yoy_chg` | One-month change in year-over-year PCE inflation |
| `pce_yoy_chg3` | Three-month change in year-over-year PCE inflation |
| `pce_yoy_ma3` | Three-month trailing average of year-over-year PCE inflation |
| `pce_yoy_ma6` | Six-month trailing average of year-over-year PCE inflation |
| `abs_inflation_gap` | Absolute distance between PCE inflation and the 2% target |

### Labour features

| Feature | Definition |
|---|---|
| `unemployment` | Latest aligned U-3 unemployment rate |
| `unemp_chg` | One-month unemployment-rate change |
| `unemp_chg3` | Three-month unemployment-rate change |
| `unemp_ma3` | Three-month trailing unemployment average |
| `natural_unemployment` | Latest aligned CBO natural-rate estimate |

### Bond-market features

Each value is the latest non-missing FRED observation **strictly before** the
meeting date. Exact meeting-date matches are disabled because a daily close can
occur after an FOMC announcement. An observation older than
`BOND_FEATURE_MAX_AGE_DAYS` causes validation to fail instead of being silently
carried forward.

| Feature | Definition |
|---|---|
| `treasury_3m_pct` | Prior-close 3-month Treasury yield |
| `treasury_6m_pct` | Prior-close 6-month Treasury yield |
| `treasury_1y_pct` | Prior-close 1-year Treasury yield |
| `treasury_2y_pct` | Prior-close 2-year Treasury yield |
| `treasury_5y_pct` | Prior-close 5-year Treasury yield |
| `treasury_10y_pct` | Prior-close 10-year Treasury yield |
| `curve_10y_minus_2y_pct` | Prior-close 10-year minus 2-year spread |
| `curve_10y_minus_3m_pct` | Prior-close 10-year minus 3-month spread |
| `treasury_3m_minus_funds_pct` | 3-month Treasury yield minus the pre-meeting funds target |
| `treasury_6m_minus_funds_pct` | 6-month Treasury yield minus the pre-meeting funds target |
| `treasury_1y_minus_funds_pct` | 1-year Treasury yield minus the pre-meeting funds target |
| `treasury_2y_minus_funds_pct` | 2-year Treasury yield minus the pre-meeting funds target |
| `treasury_5y_minus_funds_pct` | 5-year Treasury yield minus the pre-meeting funds target |
| `treasury_10y_minus_funds_pct` | 10-year Treasury yield minus the pre-meeting funds target |

### Meeting-cycle features

Every target-derived history field is shifted so it contains only information
from meetings strictly before the row being predicted.

| Feature | Definition |
|---|---|
| `is_scheduled` | `1` for a scheduled meeting and `0` for an unscheduled decision |
| `prior_decision` | Previous decision encoded as cut `-1`, hold `0`, hike `1` |
| `prior_is_change` | Whether the immediately previous meeting changed the target |
| `prior2_is_change` | Whether the meeting two decisions earlier changed the target |
| `prior3_change_count` | Number of changes across the previous three meetings |
| `prior3_direction` | Sum of signed directions across the previous three meetings |
| `prior_rate_change_bps` | Size and direction of the previous meeting's action |
| `same_direction_streak` | Consecutive prior non-hold decisions in the same direction |
| `days_since_prior_meeting` | Calendar days since the previous recorded decision |
| `days_since_prior_change` | Calendar days since the previous change, capped at 3,650 |

`features.py` also calculates interpretable context fields such as
`inflation_gap`, `labour_gap`, `real_rate_proxy`, `hawk_dove_score`, and
`policy_tightness` during feature construction. Exact or strongly redundant
linear composites are intentionally omitted from `FEATURE_COLUMNS`; they do
not enter the current estimator.

Rows missing any configured feature are reported and removed only after all
feature groups have been calculated.

## Model design

### Chronological evaluation

The newest `MODEL_TEST_FRACTION` of meetings is reserved as the final holdout.
There is no shuffling. Every required class must appear in both train and test
partitions.

Training uses class-complete `TimeSeriesSplit` folds. Each validation block is
later than its corresponding training prefix. `StandardScaler` is inside each
scikit-learn pipeline, so it is fitted separately within each model fit rather
than on the full dataset.

### Hierarchical probabilities

The two logistic regressions estimate:

```text
p_change = P(change)
p_cut_given_change = P(cut | change)
```

The three final class probabilities are coherent by construction:

```text
P(cut)  = p_change * p_cut_given_change
P(hike) = p_change * (1 - p_cut_given_change)
P(hold) = 1 - p_change
```

They are validated to sum to one for every prediction row.

### Hyperparameter and policy selection

The project searches configured regularization values and class weights using
forward training folds. The cut-versus-hike component is trained only on rows
where an actual policy change occurred.

Forward out-of-fold training probabilities are then used to select:

- the normal change threshold;
- the conditional cut-versus-hike threshold; and
- the three obvious-cut override gates.

Policy candidates are ranked primarily by three-class macro F1, followed by cut
recall, balanced accuracy, and accuracy. Among primary-model candidates within
`MODEL_SELECTION_SCORE_TOLERANCE` of the best macro F1, stronger regularization
is preferred to reduce overfitting.

The final chronological holdout is not used to choose these values.

### Auditable obvious-cut rule

The raw policy predicts a change when `p_change` crosses its selected threshold.
If it predicts a change, the conditional direction threshold chooses cut or
hike. Otherwise the raw decision is hold.

A decision is forced to cut only if all three selected gates pass:

```text
p_change >= minimum change gate
p_cut_given_change >= minimum conditional cut gate
P(cut) >= minimum joint cut gate
```

The rule does not guarantee that an override will occur. If no held-out meeting
passes every gate, the override count is correctly zero. `predictions.csv`
records the raw decision, final decision, signal, trigger, and reason.

## Generated artifacts

All generated data is ignored by Git except for `.gitkeep` placeholders.

### Raw data: `contents/data/raw/`

| File | Contents |
|---|---|
| `target_rate.csv` | Legacy daily point target |
| `target_lower.csv` | Daily target-range lower bound |
| `target_upper.csv` | Daily target-range upper bound |
| `pce_index.csv` | Monthly PCE index observations |
| `unemployment.csv` | Monthly unemployment observations |
| `natural_unemployment.csv` | Quarterly NROU observations |
| `<market_logical_name>.csv` | One raw file for each of the 16 Treasury, TIPS, breakeven, curve, and corporate-spread series |
| `fomc_meetings.csv` | Decision date, scheduled flag, and source URL |
| `source_scrape.jsonl` | Optional detailed coverage scrape |
| `source_scrape_summary.csv` | Optional compact scrape summary |

FRED raw files retain the standard `date,value` schema. Logical renaming occurs
inside the cleaning stage.

### Clean data: `contents/data/clean/`

`clean_panel.csv` contains aligned macro values, audit/reference dates, policy
states before and after each meeting, basis-point changes, and labels.

`feature_panel.csv` has the exact modeling schema:

```text
meeting_date + FEATURE_COLUMNS + is_change + decision
```

### Model outputs: `contents/outputs/`

#### `metrics.json`

Contains:

- dataset date ranges and train/test row counts;
- methodology and known vintage limitation;
- tuning scores and selected parameters;
- binary hold/change metrics;
- conditional cut/hike metrics;
- final and raw three-class policy metrics;
- confusion matrices, class counts, precision, recall, F1, and calibration metrics;
- selected thresholds and training policy audit; and
- obvious-cut override counts.

#### `coefficients.csv`

Contains standardized logistic-regression coefficients and corresponding odds
ratios for both binary component models. These are conditional associations,
not causal effects or standalone feature importance.

#### `predictions.csv`

Contains one row per meeting in the chronological holdout, including:

- actual and predicted binary decisions;
- actual, raw, and final three-class decisions;
- `P(change)` and conditional direction probabilities;
- joint cut, hold, and hike probabilities;
- obvious-cut signal and override flags; and
- an override reason when the raw decision was changed.

To print the latest saved headline metrics without additional dependencies:

```bash
python3 - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("outputs/metrics.json").read_text())
binary = metrics["models"]["is_change"]["holdout"]
decision = metrics["models"]["decision"]["holdout"]
print("Change accuracy:", f"{binary['accuracy']:.1%}")
print("Change balanced accuracy:", f"{binary['balanced_accuracy']:.1%}")
print("Three-class accuracy:", f"{decision['accuracy']:.1%}")
print("Three-class balanced accuracy:", f"{decision['balanced_accuracy']:.1%}")
PY
```

## Accuracy report

After running `model.py`, generate the diagnostic report:

```bash
python3 build_accuracy_report.py
```

The builder creates:

| Artifact | Purpose |
|---|---|
| `model_accuracy_diagnostics.ipynb` | Executed, reproducible metric and error diagnostics |
| `model_accuracy_artifact.json` | Canonical report data, chart definitions, narrative, and sources |
| `model_accuracy_report.html` | Self-contained portable HTML report with regenerated charts |

The notebook is executed with the same Python interpreter that runs the report
builder. Any notebook cell error stops the build.

The HTML renderer validates the artifact, regenerates chart SVGs, and performs
portable-report checks. It requires Node.js and the Data Analytics portable
report renderer. The builder searches the local installed plugin cache. If
automatic discovery is unavailable, set either the plugin root or renderer
script explicitly:

```bash
export DATA_ANALYTICS_REPORT_RENDERER=/path/to/data-analytics-plugin
python3 build_accuracy_report.py
```

The HTML is deterministic: if the model artifacts are unchanged, its content
may have the same hash even though the file was regenerated. The charts change
when the data in `metrics.json` or `predictions.csv` changes.

## Prediction visualizer

After `model.py` has produced `predictions.csv`, run:

```bash
python3 build_prediction_visualizer.py
```

This generates and executes:

```text
contents/outputs/prediction_visualizer.ipynb
```

The notebook contains three reader-facing visualizations:

1. **Interest-rate path:** the official post-meeting target midpoint, with
   markers for predicted cut, hold, and hike decisions.
2. **Unemployment path:** aligned unemployment and the CBO natural-rate
   estimate, with the same prediction markers.
3. **Three-class confusion matrix:** a 3×3 color-scale matrix whose rows are
   actual cut/hold/hike decisions, columns are predicted decisions, and every
   square displays the exact number of meetings.

Marker shape reinforces marker color, and incorrect class predictions receive
a dark open ring. The time-series markers visualize the predicted decision
class at the observed economic value; they are not predictions of the numeric
interest-rate or unemployment level.

The builder validates source schemas, probability sums, unique meeting dates,
the confusion-matrix total, and all notebook cell outputs. It refuses to create
a partially executed notebook and tells you to run the upstream pipeline when
the required artifacts are missing.

## Supplementary scraping

Run the optional coverage scraper directly:

```bash
python3 scrape.py
```

For each configured page, it:

- validates the source declaration;
- consults `robots.txt` using the configured user agent;
- performs a bounded request;
- records title and headings;
- records small, size-limited HTML table previews;
- preserves failed requests as records with an error message; and
- writes JSONL and summary CSV atomically.

Scraping can be enabled inside the full pipeline by setting:

```python
PIPELINE_SCRAPE_COVERAGE = True
```

Coverage scraping can be slow, blocked by third-party sites, or affected by
markup changes. Its success or failure does not change the structured model
panel.

`check_links.py` is currently an instructional scaffold. Its functions raise
`NotImplementedError`; do not include it in automated runs until its TODOs are
implemented.

## Validation and leakage controls

The pipeline fails loudly when a data contract is violated.

### Acquisition checks

- required FRED JSON fields;
- parseable dates and numeric values;
- duplicate observation dates;
- valid FOMC year range;
- duplicate meeting dates; and
- required calendar fields.

### Clean-panel checks

- sorted, unique meeting dates;
- official policy-regime splice integrity;
- macro and policy reference chronology;
- plausible value ranges;
- target arithmetic and basis-point conversion;
- agreement between `decision` and `is_change`; and
- pre/post policy observations within the accepted date window.

### Feature checks

- exact configured schema and order;
- numeric, finite, non-constant features;
- strict lags for target-derived meeting history;
- bond observations strictly before each meeting and no more than the configured age limit;
- no known post-meeting label fields in `FEATURE_COLUMNS`;
- valid decision classes and binary targets; and
- arithmetic identities such as the absolute inflation gap.

### Model and artifact checks

- chronological, non-overlapping train/test partitions;
- class-complete training prefixes;
- scaler fitting inside model pipelines;
- probability bounds and sums;
- binary/three-class prediction agreement;
- confusion-matrix and metric reproduction;
- atomic JSON and CSV output writes;
- executed notebook cells with no saved errors; and
- validated, self-contained HTML report generation.

## Known limitations

### Not point-in-time macro data

The largest methodological limitation is vintage leakage risk. FRED observation
dates identify economic reference periods, not the exact date a value was first
released or the value known in real time. FRED series can also be revised.

For a production-quality historical backtest, replace current revised inputs
with ALFRED vintages or another release-aware dataset and define a fixed
prediction cutoff, such as the market close one day before each meeting.

### Small and imbalanced sample

The FOMC meets only several times per year. Holds dominate, while cuts and hikes
are concentrated in policy cycles. Class-level recall and headline accuracy can
change substantially with only a few meetings.

### Regime dependence

The sample spans point-target and target-range regimes, the zero-lower-bound
period, emergency decisions, tightening cycles, and easing cycles. A relationship
learned in one regime may not transfer to another.

### Limited information set

The current model does not include:

- historical fed-funds-futures or market-implied meeting probabilities;
- survey consensus available before each meeting;
- real-time payroll, GDP, broader financial-condition, or inflation-release vintages;
- FOMC communication text transformed into timestamped features; or
- a persisted production estimator and live next-meeting inference command.

### Optional web coverage is not structured evidence

Titles, headings, and table previews are unsuitable as model inputs without
additional definitions and timestamp controls. Broad source coverage should not
be mistaken for a broad predictive information set.

## Troubleshooting

### `Missing FRED API key`

Confirm that `contents/.env` contains:

```dotenv
FRED_API_KEY=your_key
```

Run from the project environment and do not name the variable `API_KEY`; the
configured name is `FRED_API_KEY`.

### FRED returns HTTP 400

Common causes include an absent/invalid API key, a misspelled series ID, or an
invalid date. Test a request using the same parameters constructed by
`fetch_fred_series`. Do not print or commit the actual key.

### A raw CSV is missing

Run:

```bash
python3 pull_from_apis.py
```

or rerun the complete `pipeline.py`. `clean.py` deliberately refuses to invent
missing inputs.

### A series appears to start later than requested

The configured calendar start is not necessarily the first usable model row.
The final start date is constrained by overlap among all series, the policy
regime splice, twelve months needed for PCE inflation, rolling windows, meeting
alignment, and complete feature requirements.

### The FOMC parser finds no meetings

The official page markup may have changed. Inspect the current and historical
Federal Reserve calendar HTML, update the selectors in
`fetch_fomc_meeting_calendar`, and validate against saved fixtures before
trusting the revised parser.

### The notebook uses the wrong Python environment

Run the report builder with the intended interpreter:

```bash
../.venv/bin/python build_accuracy_report.py
```

The builder prepends that interpreter's directory when launching the notebook
kernel and rejects saved cell errors.

### The HTML report is not updated

Current versions of `build_accuracy_report.py` invoke the renderer and print:

```text
Saved portable HTML report to .../model_accuracy_report.html
HTML verification: passed (... charts, ... tables)
```

If the renderer cannot be discovered, set `DATA_ANALYTICS_REPORT_RENDERER`.
Remember that unchanged model data produces identical chart content.

### Coverage pages fail to scrape

Failures are expected when robots rules disallow access, a site blocks automated
requests, or markup changes. Check `source_scrape_summary.csv`. Coverage failures
do not invalidate the structured FRED model data.

## Extending the project

### Add a structured model series

1. Declare the logical name, FRED ID, frequency, role, and description in
   `FRED_SERIES`.
2. Rerun acquisition and confirm the standard `date,value` raw schema.
3. Add explicit cleaning and alignment logic.
4. Preserve a reference date and, ideally, release/vintage timestamp.
5. Add a feature declaration to `FEATURE_COLUMNS` only after defining its
   transformation and leakage test.
6. Update validators, the report, and this README.

Adding a `SUPPLEMENTARY_SOURCES` entry alone does not add a model feature.

### Add market expectations or consensus

Define exactly which value would have been observable at the prediction cutoff.
For futures-derived probabilities, store the contract, settlement timestamp,
meeting mapping, calculation method, and source license. For surveys, store the
survey field date, publication date, consensus statistic, and revision policy.

### Add live next-meeting inference

A production inference stage would need to:

1. persist the fitted scaler and both classifiers;
2. construct a feature row at a fixed real-time cutoff;
3. reject any unavailable or post-cutoff values;
4. apply the saved—not newly tuned—threshold policy;
5. log input vintages, probabilities, and final decision; and
6. distinguish a forecast from historical holdout predictions.

### Development smoke checks

From `contents/`:

```bash
python3 -m py_compile \
  config.py pull_from_apis.py scrape.py clean.py \
  features.py model.py pipeline.py build_accuracy_report.py

python3 clean.py
python3 features.py
python3 model.py
python3 build_accuracy_report.py
```

The last four commands assume their upstream generated files already exist.
Use `pipeline.py` when a complete refresh is required.

## Reproducibility notes

- `RANDOM_STATE` is fixed in `config.py`.
- Train/test order is chronological.
- Cross-validation is forward-only.
- Generated CSV and JSON model artifacts use atomic replacement.
- The HTML report is self-contained and rendered from the canonical JSON
  artifact.
- Data and outputs are intentionally ignored by Git, so exact reproduction
  requires rerunning acquisition at a documented date or archiving immutable
  source snapshots separately.

No license file is currently included. Add one before distributing or reusing
the project under explicit licensing terms.
