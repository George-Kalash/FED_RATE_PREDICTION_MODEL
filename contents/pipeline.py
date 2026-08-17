"""Four-stage orchestration declaration for the Fed decision predictor.

Dependency flow:

``acquire -> clean/align -> engineer features -> train``

Supplementary coverage scraping is part of acquisition for documentation, but
its output is deliberately not a dependency of cleaning or modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from clean import build_clean_panel
from config import (
    CLEAN_PANEL_PATH,
    DATA_RAW,
    FEATURE_PANEL_PATH,
    FOMC_CALENDAR_URL,
    MODEL_COEFFICIENTS_PATH,
    MODEL_METRICS_PATH,
    MODEL_PREDICTIONS_PATH,
    OUTPUTS,
    PIPELINE_END_YEAR,
    PIPELINE_FRED_OBSERVATION_START,
    PIPELINE_SCRAPE_COVERAGE,
    PIPELINE_START_YEAR,
    SOURCE_SCRAPE_JSONL_PATH,
    SOURCE_SCRAPE_SUMMARY_PATH,
)
from features import build_feature_panel
from model import train_models
from pull_from_apis import (
    fetch_fomc_meeting_calendar,
    pull_all_model_series,
    resolve_fred_api_key,
    save_fomc_meeting_calendar,
)
from scrape import scrape_all_sources, write_jsonl, write_summary_csv


@dataclass(frozen=True)
class PipelinePaths:
    """All filesystem handoffs between pipeline stages."""

    raw_dir: Path
    source_scrape_jsonl: Path
    source_scrape_summary: Path
    clean_panel: Path
    feature_panel: Path
    output_dir: Path


def default_paths() -> PipelinePaths:
    """Return the immutable filesystem handoffs declared in ``config.py``."""
    return PipelinePaths(
        raw_dir=DATA_RAW,
        source_scrape_jsonl=SOURCE_SCRAPE_JSONL_PATH,
        source_scrape_summary=SOURCE_SCRAPE_SUMMARY_PATH,
        clean_panel=CLEAN_PANEL_PATH,
        feature_panel=FEATURE_PANEL_PATH,
        output_dir=OUTPUTS,
    )


def run_acquisition() -> None:
    """Refresh structured inputs and optionally coverage-only artifacts."""
    api_key = resolve_fred_api_key()
    series_paths = pull_all_model_series(
        api_key,
        observation_start=PIPELINE_FRED_OBSERVATION_START,
    )
    calendar = fetch_fomc_meeting_calendar(
        FOMC_CALENDAR_URL,
        start_year=PIPELINE_START_YEAR,
        end_year=PIPELINE_END_YEAR,
    )
    calendar_path = save_fomc_meeting_calendar(calendar)

    print(
        f"Acquisition: saved {len(series_paths)} FRED series and "
        f"{len(calendar)} FOMC meetings to {DATA_RAW}"
    )
    for logical_name, path in series_paths.items():
        if not path.is_file():
            raise RuntimeError(f"Acquisition did not create {logical_name}: {path}")
    if not calendar_path.is_file():
        raise RuntimeError(f"Acquisition did not create calendar: {calendar_path}")

    if PIPELINE_SCRAPE_COVERAGE:
        coverage_records = scrape_all_sources()
        jsonl_path = write_jsonl(coverage_records)
        summary_path = write_summary_csv(coverage_records)
        succeeded = sum(record["error"] is None for record in coverage_records)
        print(
            "Coverage-only scrape: "
            f"{succeeded}/{len(coverage_records)} succeeded; "
            f"saved {jsonl_path} and {summary_path}"
        )
    else:
        print("Coverage-only scrape: skipped by config (not a model dependency)")


def run_pipeline() -> None:
    """Run acquisition, cleaning, features, and training in dependency order."""
    paths = default_paths()
    print(
        f"Pipeline configuration: FOMC years {PIPELINE_START_YEAR}-"
        f"{PIPELINE_END_YEAR}; raw data {paths.raw_dir}"
    )

    run_acquisition()

    clean_path = build_clean_panel()
    clean_rows = len(pd.read_csv(clean_path))
    print(f"Cleaning: saved {clean_rows} meeting rows to {clean_path}")

    feature_path = build_feature_panel()
    feature_frame = pd.read_csv(feature_path)
    print(
        f"Features: saved {len(feature_frame)} rows and "
        f"{len(feature_frame.columns) - 3} features to {feature_path}"
    )

    metrics_path, coefficients_path, predictions_path = train_models()
    if (
        metrics_path != MODEL_METRICS_PATH
        or coefficients_path != MODEL_COEFFICIENTS_PATH
        or predictions_path != MODEL_PREDICTIONS_PATH
    ):
        raise RuntimeError("Training returned paths that disagree with config.py")
    print(
        "Training: saved "
        f"{metrics_path}, {coefficients_path}, and {predictions_path}"
    )
    print("Pipeline complete")


def main() -> None:
    """Run the configured pipeline without command-line arguments."""
    run_pipeline()


if __name__ == "__main__":
    main()
