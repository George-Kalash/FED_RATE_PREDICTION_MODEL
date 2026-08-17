"""Central declarations for the Federal Reserve decision pipeline.

Keep configuration here and implementation in the stage modules:

1. ``pull_from_apis.py`` and ``scrape.py`` acquire raw inputs.
2. ``clean.py`` aligns observations and creates decision labels.
3. ``features.py`` creates information available before each meeting.
4. ``model.py`` trains and evaluates logistic-regression models.

Only the structured series in ``FRED_SERIES`` and the FOMC meeting calendar
may enter the modeling panel. ``SUPPLEMENTARY_SOURCES`` is coverage metadata
and must never be merged into the training data unless a future feature has a
documented definition, release timestamp, and leakage test.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Final, Literal, TypedDict

from dotenv import load_dotenv


class FredSeriesSpec(TypedDict):
    """Declaration for one structured FRED input."""

    series_id: str
    frequency: Literal["daily", "monthly", "quarterly"]
    role: Literal["policy", "inflation", "labour"]
    description: str


class WebSourceSpec(TypedDict):
    """Declaration for a supplementary page that is not a model input."""

    id: str
    category: str
    url: str


# Project paths. The data folders already live beside these Python modules.
PROJECT_DIR: Final = Path(__file__).resolve().parent
DOTENV_PATH: Final = PROJECT_DIR / ".env"
DATA_DIR: Final = PROJECT_DIR / "data"
DATA_RAW: Final = DATA_DIR / "raw"
DATA_PROCESSED: Final = DATA_DIR / "clean"
FOMC_MEETINGS_PATH: Final = DATA_RAW / "fomc_meetings.csv"
CLEAN_PANEL_PATH: Final = DATA_PROCESSED / "clean_panel.csv"
FEATURE_PANEL_PATH: Final = DATA_PROCESSED / "feature_panel.csv"
SOURCE_SCRAPE_JSONL_PATH: Final = DATA_RAW / "source_scrape.jsonl"
SOURCE_SCRAPE_SUMMARY_PATH: Final = DATA_RAW / "source_scrape_summary.csv"
OUTPUTS: Final = PROJECT_DIR / "outputs"
MODEL_METRICS_PATH: Final = OUTPUTS / "metrics.json"
MODEL_COEFFICIENTS_PATH: Final = OUTPUTS / "coefficients.csv"
MODEL_PREDICTIONS_PATH: Final = OUTPUTS / "predictions.csv"

# Network configuration. Never put the actual API key in this file.
FRED_API_KEY_ENV: Final = "FRED_API_KEY"
load_dotenv(DOTENV_PATH)
FRED_API_KEY: Final[str | None] = os.environ.get(FRED_API_KEY_ENV)
FRED_API_BASE_URL: Final = (
    "https://api.stlouisfed.org/fred/series/observations"
)
FOMC_CALENDAR_URL: Final = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
USER_AGENT: Final = (
    "FedDecisionPredictor/0.1 (educational research; contact: add-email-here)"
)
REQUEST_TIMEOUT_SECONDS: Final = 25

# End-to-end pipeline behavior. Coverage scraping remains optional because it
# does not feed the model and can be slow or blocked by third-party websites.
PIPELINE_START_YEAR: Final = 1998
PIPELINE_END_YEAR: Final = date.today().year
PIPELINE_FRED_OBSERVATION_START: Final[str | None] = None
PIPELINE_SCRAPE_COVERAGE: Final = False

# Supplementary coverage scraping. These limits keep the audit artifact small;
# scraped content is never an approved model input.
SCRAPE_DELAY_SECONDS: Final = 0.5
SCRAPE_MAX_TITLE_CHARS: Final = 300
SCRAPE_MAX_HEADINGS: Final = 40
SCRAPE_MAX_HEADING_CHARS: Final = 300
SCRAPE_MAX_TABLES: Final = 5
SCRAPE_MAX_TABLE_ROWS: Final = 5
SCRAPE_MAX_TABLE_COLUMNS: Final = 12
SCRAPE_MAX_CELL_CHARS: Final = 500

# Structured modeling backbone. PCEPI matches the Fed's inflation objective;
# UNRATE and NROU allow the labour gap to vary rather than assuming 6 percent.
# The legacy point target is used before December 2008; afterward the policy
# level is derived from the target-range midpoint.
FRED_SERIES: Final[dict[str, FredSeriesSpec]] = {
    "target_rate": {
        "series_id": "DFEDTAR",
        "frequency": "daily",
        "role": "policy",
        "description": "Federal funds target rate before the target-range regime",
    },
    "target_upper": {
        "series_id": "DFEDTARU",
        "frequency": "daily",
        "role": "policy",
        "description": "Federal funds target range upper limit (percent)",
    },
    "target_lower": {
        "series_id": "DFEDTARL",
        "frequency": "daily",
        "role": "policy",
        "description": "Federal funds target range lower limit (percent)",
    },
    "pce_index": {
        "series_id": "PCEPI",
        "frequency": "monthly",
        "role": "inflation",
        "description": "Personal Consumption Expenditures price index",
    },
    "unemployment": {
        "series_id": "UNRATE",
        "frequency": "monthly",
        "role": "labour",
        "description": "Civilian unemployment rate, seasonally adjusted",
    },
    "natural_unemployment": {
        "series_id": "NROU",
        "frequency": "quarterly",
        "role": "labour",
        "description": "CBO estimate of the natural unemployment rate",
    },
}

# Model and label declarations.
DATE_COLUMN: Final = "date"
MEETING_DATE_COLUMN: Final = "meeting_date"
INFLATION_TARGET_PERCENT: Final = 2.0
RATE_CHANGE_TOLERANCE_BPS: Final = 0.5
DECISION_CLASSES: Final[tuple[str, ...]] = ("cut", "hold", "hike")
POLICY_REGIMES: Final[tuple[str, ...]] = ("point_target", "target_range")
PRIMARY_TARGET: Final = "is_change"
DIAGNOSTIC_TARGET: Final = "decision"
DIRECTION_TARGET: Final = "direction"
RANDOM_STATE: Final = 42
CV_SPLITS: Final = 5
LOGISTIC_C_VALUES: Final[tuple[float, ...]] = (0.1, 0.5, 1.0, 2.0)
MODEL_TEST_FRACTION: Final = 0.2
LOGISTIC_MAX_ITERATIONS: Final = 5_000
PRIMARY_MODEL_SCORING: Final = "balanced_accuracy"
DIAGNOSTIC_MODEL_SCORING: Final = "f1_macro"

# Hierarchical model and decision-policy search. Every value is selected using
# chronological training folds; the final holdout never chooses a threshold.
PRIMARY_CLASS_WEIGHT_OPTIONS: Final[tuple[str | None, ...]] = (None, "balanced")
DIRECTION_CUT_WEIGHT_OPTIONS: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 3.0)
CHANGE_THRESHOLD_VALUES: Final[tuple[float, ...]] = (0.35, 0.45, 0.5, 0.55, 0.65)
DIRECTION_CUT_THRESHOLD_VALUES: Final[tuple[float, ...]] = (0.4, 0.5, 0.6)
CUT_OVERRIDE_MIN_CHANGE_VALUES: Final[tuple[float, ...]] = (0.25, 0.35, 0.45)
CUT_OVERRIDE_DIRECTION_VALUES: Final[tuple[float, ...]] = (0.75, 0.85, 0.9)
CUT_OVERRIDE_JOINT_VALUES: Final[tuple[float, ...]] = (0.2, 0.25, 0.3)
DECISION_POLICY_SCORING: Final = "f1_macro"
MODEL_SELECTION_SCORE_TOLERANCE: Final = 0.02

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "rate_level",
    "rate_chg_1m",
    "rate_chg_3m",
    "pce_yoy",
    "pce_yoy_chg",
    "pce_yoy_chg3",
    "pce_yoy_ma3",
    "pce_yoy_ma6",
    "unemployment",
    "unemp_chg",
    "unemp_chg3",
    "unemp_ma3",
    "natural_unemployment",
    "abs_inflation_gap",
    "is_scheduled",
    "prior_decision",
    "prior_is_change",
    "prior2_is_change",
    "prior3_change_count",
    "prior3_direction",
    "prior_rate_change_bps",
    "same_direction_streak",
    "days_since_prior_meeting",
    "days_since_prior_change",
)

# Feature definitions. Windows are calendar months, never prior FOMC rows.
RATE_CHANGE_WINDOWS_MONTHS: Final[tuple[int, ...]] = (1, 3)
INFLATION_CHANGE_WINDOWS_MONTHS: Final[tuple[int, ...]] = (1, 3)
INFLATION_AVERAGE_WINDOWS_MONTHS: Final[tuple[int, ...]] = (3, 6)
UNEMPLOYMENT_CHANGE_WINDOWS_MONTHS: Final[tuple[int, ...]] = (1, 3)
UNEMPLOYMENT_AVERAGE_WINDOWS_MONTHS: Final[tuple[int, ...]] = (3,)
HAWK_DOVE_LABOUR_WEIGHT: Final = 0.5

# Coverage-only pages. These can produce source_scrape.jsonl and a summary,
# but their titles, headings, and tables are not approved model features.
SUPPLEMENTARY_SOURCES: Final[list[WebSourceSpec]] = [
    {"id": "fed_home", "category": "central_bank", "url": "https://www.federalreserve.gov/"},
    {"id": "fed_monetary_policy", "category": "rates", "url": "https://www.federalreserve.gov/monetarypolicy.htm"},
    {"id": "fed_fomc", "category": "rates", "url": "https://www.federalreserve.gov/monetarypolicy/fomc.htm"},
    {"id": "fed_calendars", "category": "rates", "url": FOMC_CALENDAR_URL},
    {"id": "fed_statements", "category": "rates", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"},
    {"id": "fed_minutes", "category": "rates", "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm"},
    {"id": "fed_press", "category": "communications", "url": "https://www.federalreserve.gov/newsevents/pressreleases.htm"},
    {"id": "fed_speeches", "category": "communications", "url": "https://www.federalreserve.gov/newsevents/speeches.htm"},
    {"id": "fed_testimony", "category": "communications", "url": "https://www.federalreserve.gov/newsevents/testimony.htm"},
    {"id": "fed_data", "category": "macro", "url": "https://www.federalreserve.gov/data.htm"},
    {"id": "fed_financial_stability", "category": "financial_stability", "url": "https://www.federalreserve.gov/publications/financial-stability-report.htm"},
    {"id": "fed_beige_book", "category": "macro", "url": "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm"},
    {"id": "bls_home", "category": "labour", "url": "https://www.bls.gov/"},
    {"id": "bls_cpi", "category": "inflation", "url": "https://www.bls.gov/cpi/"},
    {"id": "bls_employment", "category": "labour", "url": "https://www.bls.gov/ces/"},
    {"id": "bls_labor_force", "category": "labour", "url": "https://www.bls.gov/cps/"},
    {"id": "bea_home", "category": "macro", "url": "https://www.bea.gov/"},
    {"id": "bea_pce", "category": "inflation", "url": "https://www.bea.gov/data/personal-consumption-expenditures-price-index"},
    {"id": "bea_gdp", "category": "macro", "url": "https://www.bea.gov/data/gdp/gross-domestic-product"},
    {"id": "treasury_home", "category": "fiscal", "url": "https://home.treasury.gov/"},
    {"id": "treasury_data", "category": "markets", "url": "https://home.treasury.gov/resource-center/data-chart-center"},
    {"id": "treasury_rates", "category": "markets", "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates"},
    {"id": "cbo_home", "category": "fiscal", "url": "https://www.cbo.gov/"},
    {"id": "cbo_economy", "category": "macro", "url": "https://www.cbo.gov/topics/economy"},
    {"id": "fdic_home", "category": "banking", "url": "https://www.fdic.gov/"},
    {"id": "fdic_data", "category": "banking", "url": "https://www.fdic.gov/analysis/"},
    {"id": "occ_home", "category": "banking", "url": "https://www.occ.gov/"},
    {"id": "sec_home", "category": "markets", "url": "https://www.sec.gov/"},
    {"id": "finra_home", "category": "markets", "url": "https://www.finra.org/"},
    {"id": "fred_home", "category": "macro", "url": "https://fred.stlouisfed.org/"},
    {"id": "fred_federal_funds", "category": "rates", "url": "https://fred.stlouisfed.org/tags/series?t=federal+funds"},
    {"id": "fred_pce", "category": "inflation", "url": "https://fred.stlouisfed.org/tags/series?t=pce"},
    {"id": "fred_unemployment", "category": "labour", "url": "https://fred.stlouisfed.org/tags/series?t=unemployment"},
    {"id": "new_york_fed", "category": "central_bank", "url": "https://www.newyorkfed.org/"},
    {"id": "new_york_fed_markets", "category": "markets", "url": "https://www.newyorkfed.org/markets"},
    {"id": "atlanta_fed", "category": "central_bank", "url": "https://www.atlantafed.org/"},
    {"id": "atlanta_fed_gdpnow", "category": "macro", "url": "https://www.atlantafed.org/cqer/research/gdpnow"},
    {"id": "cleveland_fed", "category": "central_bank", "url": "https://www.clevelandfed.org/"},
    {"id": "cleveland_fed_inflation", "category": "inflation", "url": "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"},
    {"id": "st_louis_fed", "category": "central_bank", "url": "https://www.stlouisfed.org/"},
    {"id": "imf_us", "category": "macro", "url": "https://www.imf.org/en/Countries/USA"},
    {"id": "oecd_us", "category": "macro", "url": "https://www.oecd.org/unitedstates/"},
    {"id": "world_bank_us", "category": "macro", "url": "https://data.worldbank.org/country/united-states"},
]

assert len(SUPPLEMENTARY_SOURCES) >= 40
