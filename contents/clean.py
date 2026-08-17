"""Stage 2: clean, align, and label Fed decision observations.

The intended training grain is one row per FOMC decision date. Macroeconomic
values must represent information available before that decision. The future
policy-rate movement is permitted only as the label, never as a feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config import (
    CLEAN_PANEL_PATH,
    DATA_RAW,
    DECISION_CLASSES,
    POLICY_REGIMES,
    RATE_CHANGE_TOLERANCE_BPS,
)


Decision = Literal["cut", "hold", "hike"]


def load_raw_series(path: Path, value_name: str) -> "pd.DataFrame":
    """Load one standard ``date,value`` CSV and rename its value column.

    The input must contain exactly ``date`` and ``value`` columns. Missing numeric
    values are preserved, while non-empty, non-numeric values are rejected.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw series file does not exist: {path}")
    if not isinstance(value_name, str) or not value_name.strip():
        raise ValueError("value_name must be a non-empty string")
    if value_name == "date":
        raise ValueError("value_name cannot be 'date'")

    frame = pd.read_csv(path)
    expected_columns = {"date", "value"}
    actual_columns = set(frame.columns)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(
            f"Raw series {path} must contain exactly date,value columns; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if frame.empty:
        raise ValueError(f"Raw series {path} contains no observations")

    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    invalid_dates = frame["date"].notna() & parsed_dates.isna()
    if invalid_dates.any() or parsed_dates.isna().any():
        bad_rows = frame.index[parsed_dates.isna()].tolist()[:5]
        raise ValueError(
            f"Raw series {path} contains missing or invalid dates at rows {bad_rows}"
        )

    raw_values = frame["value"]
    parsed_values = pd.to_numeric(raw_values, errors="coerce")
    nonempty_values = raw_values.notna() & raw_values.astype("string").str.strip().ne("")
    invalid_values = nonempty_values & parsed_values.isna()
    if invalid_values.any():
        bad_rows = frame.index[invalid_values].tolist()[:5]
        raise ValueError(
            f"Raw series {path} contains non-numeric values at rows {bad_rows}"
        )

    frame = pd.DataFrame({"date": parsed_dates, value_name: parsed_values})
    duplicate_dates = frame.loc[frame["date"].duplicated(keep=False), "date"]
    if not duplicate_dates.empty:
        duplicate_text = ", ".join(
            value.strftime("%Y-%m-%d") for value in duplicate_dates.drop_duplicates()
        )
        raise ValueError(
            f"Raw series {path} contains duplicate dates: {duplicate_text}"
        )

    return frame.sort_values("date").reset_index(drop=True)


def build_policy_rate_series(
    target_rate: "pd.DataFrame",
    target_lower: "pd.DataFrame",
    target_upper: "pd.DataFrame",
) -> "pd.DataFrame":
    """Build one daily official policy series across both target regimes.

    Before the first date with both range bounds, the legacy point target is
    represented with identical lower and upper bounds. From the range regime's
    first complete date onward, ``policy_rate`` is the range midpoint. The
    explicit ``policy_regime`` column makes the splice auditable and prevents an
    effective-market rate from being silently substituted for an official target.
    """
    expected_inputs = (
        ("target_rate", target_rate),
        ("target_lower", target_lower),
        ("target_upper", target_upper),
    )
    prepared: dict[str, pd.DataFrame] = {}

    for value_column, frame in expected_inputs:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{value_column} must be a pandas DataFrame")

        expected_columns = {"date", value_column}
        actual_columns = set(frame.columns)
        if actual_columns != expected_columns:
            missing = sorted(expected_columns - actual_columns)
            unexpected = sorted(actual_columns - expected_columns)
            raise ValueError(
                f"{value_column} must contain exactly date,{value_column}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if frame.empty:
            raise ValueError(f"{value_column} contains no observations")

        clean_frame = frame.loc[:, ["date", value_column]].copy()
        clean_frame["date"] = pd.to_datetime(clean_frame["date"], errors="coerce")
        if clean_frame["date"].isna().any():
            raise ValueError(f"{value_column} contains missing or invalid dates")
        if clean_frame["date"].duplicated().any():
            raise ValueError(f"{value_column} contains duplicate dates")

        raw_values = clean_frame[value_column]
        clean_frame[value_column] = pd.to_numeric(raw_values, errors="coerce")
        invalid_values = raw_values.notna() & clean_frame[value_column].isna()
        if invalid_values.any():
            raise ValueError(f"{value_column} contains non-numeric values")

        prepared[value_column] = clean_frame.sort_values("date").reset_index(drop=True)

    legacy_observations = prepared["target_rate"].dropna(subset=["target_rate"])
    lower_observations = prepared["target_lower"].dropna(subset=["target_lower"])
    upper_observations = prepared["target_upper"].dropna(subset=["target_upper"])
    if legacy_observations.empty:
        raise ValueError("target_rate contains no usable observations")
    if lower_observations.empty or upper_observations.empty:
        raise ValueError("Target-range bounds contain no usable observations")

    range_start = max(
        lower_observations["date"].min(),
        upper_observations["date"].min(),
    )
    required_legacy_end = range_start - pd.Timedelta(days=1)
    if legacy_observations["date"].max() < required_legacy_end:
        raise ValueError(
            "Legacy target rate ends before the target-range regime begins"
        )

    first_date = legacy_observations["date"].min()
    last_date = max(
        lower_observations["date"].max(),
        upper_observations["date"].max(),
    )
    daily = pd.DataFrame({"date": pd.date_range(first_date, last_date, freq="D")})

    daily = daily.merge(
        prepared["target_rate"], on="date", how="left", validate="one_to_one"
    )
    daily = daily.merge(
        prepared["target_lower"], on="date", how="left", validate="one_to_one"
    )
    daily = daily.merge(
        prepared["target_upper"], on="date", how="left", validate="one_to_one"
    )
    daily[["target_rate", "target_lower", "target_upper"]] = daily[
        ["target_rate", "target_lower", "target_upper"]
    ].ffill()

    legacy_mask = daily["date"] < range_start
    range_mask = ~legacy_mask
    daily.loc[legacy_mask, "target_lower"] = daily.loc[legacy_mask, "target_rate"]
    daily.loc[legacy_mask, "target_upper"] = daily.loc[legacy_mask, "target_rate"]
    daily["policy_regime"] = np.where(
        legacy_mask,
        "point_target",
        "target_range",
    )
    if daily.loc[legacy_mask, "target_rate"].isna().any():
        raise ValueError("Legacy target regime contains gaps before the range era")
    if daily.loc[range_mask, ["target_lower", "target_upper"]].isna().any().any():
        raise ValueError("Target-range regime contains missing bounds")

    invalid_ranges = daily["target_lower"] > daily["target_upper"]
    if invalid_ranges.any():
        bad_dates = daily.loc[invalid_ranges, "date"].dt.strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(
            f"Target lower bound exceeds target upper bound on dates {bad_dates}"
        )
    if (
        daily.loc[range_mask, "target_lower"]
        >= daily.loc[range_mask, "target_upper"]
    ).any():
        raise ValueError("Target-range regime must have a positive range width")

    daily["policy_rate"] = (
        daily["target_lower"] + daily["target_upper"]
    ) / 2.0

    return daily.loc[
        :,
        [
            "date",
            "target_lower",
            "target_upper",
            "policy_rate",
            "policy_regime",
        ],
    ]


def calculate_pce_yoy(pce_index: "pd.DataFrame") -> "pd.DataFrame":
    """Convert the monthly PCE index level to 12-month percent inflation.

    Normalize observations to month-end, retain the last observation in each
    month, and calculate ``pct_change(12) * 100``. A YoY value is emitted only
    when the current and previous 12 monthly index levels are all present.
    """
    if not isinstance(pce_index, pd.DataFrame):
        raise TypeError("pce_index must be a pandas DataFrame")

    expected_columns = {"date", "pce_index"}
    actual_columns = set(pce_index.columns)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        raise ValueError(
            "pce_index must contain exactly date,pce_index; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if pce_index.empty:
        raise ValueError("pce_index contains no observations")

    monthly = pce_index.loc[:, ["date", "pce_index"]].copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    if monthly["date"].isna().any():
        raise ValueError("pce_index contains missing or invalid dates")
    if monthly["date"].duplicated().any():
        raise ValueError("pce_index contains duplicate dates")

    raw_values = monthly["pce_index"]
    monthly["pce_index"] = pd.to_numeric(raw_values, errors="coerce")
    invalid_values = raw_values.notna() & monthly["pce_index"].isna()
    if invalid_values.any():
        raise ValueError("pce_index contains non-numeric values")
    if (monthly["pce_index"].dropna() <= 0).any():
        raise ValueError("pce_index values must be greater than zero")

    monthly = monthly.sort_values("date")
    monthly = (
        monthly.set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )

    complete_window = (
        monthly["pce_index"]
        .notna()
        .rolling(window=13, min_periods=13)
        .sum()
        .eq(13)
    )
    monthly["pce_yoy"] = (
        monthly["pce_index"].pct_change(periods=12, fill_method=None) * 100.0
    ).where(complete_window)

    if not monthly["pce_yoy"].notna().any():
        raise ValueError(
            "pce_index requires at least 13 consecutive monthly observations "
            "to calculate year-over-year inflation"
        )

    return monthly.loc[:, ["date", "pce_index", "pce_yoy"]]


def prepare_monthly_labour_data(
    unemployment: "pd.DataFrame", natural_unemployment: "pd.DataFrame"
) -> "pd.DataFrame":
    """Align actual unemployment and the lower-frequency natural-rate estimate.

    Use the observed unemployment series as the monthly calendar. Quarterly NROU
    values are normalized to month-end and carried only into later months; they
    are never backfilled into months before the first available estimate. Future
    NROU projections beyond the latest unemployment observation are excluded.

    FRED observation dates describe reference periods, not historical release
    vintages. Strict point-in-time work must add release/vintage metadata later.
    """
    expected_inputs = (
        ("unemployment", unemployment),
        ("natural_unemployment", natural_unemployment),
    )
    prepared: dict[str, pd.DataFrame] = {}

    for value_column, frame in expected_inputs:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{value_column} must be a pandas DataFrame")

        expected_columns = {"date", value_column}
        actual_columns = set(frame.columns)
        if actual_columns != expected_columns:
            missing = sorted(expected_columns - actual_columns)
            unexpected = sorted(actual_columns - expected_columns)
            raise ValueError(
                f"{value_column} must contain exactly date,{value_column}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if frame.empty:
            raise ValueError(f"{value_column} contains no observations")

        clean_frame = frame.loc[:, ["date", value_column]].copy()
        clean_frame["date"] = pd.to_datetime(clean_frame["date"], errors="coerce")
        if clean_frame["date"].isna().any():
            raise ValueError(f"{value_column} contains missing or invalid dates")
        if clean_frame["date"].duplicated().any():
            raise ValueError(f"{value_column} contains duplicate dates")

        raw_values = clean_frame[value_column]
        clean_frame[value_column] = pd.to_numeric(raw_values, errors="coerce")
        nonempty_values = (
            raw_values.notna() & raw_values.astype("string").str.strip().ne("")
        )
        invalid_values = nonempty_values & clean_frame[value_column].isna()
        if invalid_values.any():
            raise ValueError(f"{value_column} contains non-numeric values")

        valid_values = clean_frame[value_column].dropna()
        if valid_values.empty:
            raise ValueError(f"{value_column} contains no numeric observations")
        if ((valid_values < 0) | (valid_values > 100)).any():
            raise ValueError(f"{value_column} values must be between 0 and 100")

        prepared[value_column] = clean_frame.sort_values("date").reset_index(drop=True)

    monthly_unemployment = (
        prepared["unemployment"]
        .set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )
    monthly_natural_rate = (
        prepared["natural_unemployment"]
        .set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )

    labour = monthly_unemployment.merge(
        monthly_natural_rate,
        on="date",
        how="left",
        validate="one_to_one",
    )
    labour["natural_unemployment"] = labour["natural_unemployment"].ffill()

    if not labour["natural_unemployment"].notna().any():
        raise ValueError(
            "unemployment and natural_unemployment have no usable overlapping dates"
        )

    return labour.loc[:, ["date", "unemployment", "natural_unemployment"]]


def build_monthly_macro_panel(
    policy_rate: "pd.DataFrame",
    pce_yoy: "pd.DataFrame",
    labour: "pd.DataFrame",
) -> "pd.DataFrame":
    """Merge rate, inflation, and labour inputs on a month-end calendar.

    Resample the daily policy target using month-end ``last`` semantics, then
    merge all inputs over a complete monthly calendar. Policy-rate values may be
    forward-filled because the target remains in force until changed. Inflation
    and unemployment values are never filled here, so missingness stays visible.
    """
    expected_schemas = (
        (
            "policy_rate",
            policy_rate,
            [
                "date",
                "target_lower",
                "target_upper",
                "policy_rate",
                "policy_regime",
            ],
        ),
        ("pce_yoy", pce_yoy, ["date", "pce_index", "pce_yoy"]),
        (
            "labour",
            labour,
            ["date", "unemployment", "natural_unemployment"],
        ),
    )
    prepared: dict[str, pd.DataFrame] = {}

    for input_name, frame, expected_columns in expected_schemas:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{input_name} must be a pandas DataFrame")
        if set(frame.columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(frame.columns))
            unexpected = sorted(set(frame.columns) - set(expected_columns))
            raise ValueError(
                f"{input_name} has an invalid schema; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if frame.empty:
            raise ValueError(f"{input_name} contains no observations")

        clean_frame = frame.loc[:, expected_columns].copy()
        clean_frame["date"] = pd.to_datetime(clean_frame["date"], errors="coerce")
        if clean_frame["date"].isna().any():
            raise ValueError(f"{input_name} contains missing or invalid dates")
        if clean_frame["date"].duplicated().any():
            raise ValueError(f"{input_name} contains duplicate dates")

        numeric_columns = [
            column
            for column in expected_columns[1:]
            if column != "policy_regime"
        ]
        for value_column in numeric_columns:
            raw_values = clean_frame[value_column]
            clean_frame[value_column] = pd.to_numeric(raw_values, errors="coerce")
            nonempty_values = (
                raw_values.notna() & raw_values.astype("string").str.strip().ne("")
            )
            if (nonempty_values & clean_frame[value_column].isna()).any():
                raise ValueError(
                    f"{input_name}.{value_column} contains non-numeric values"
                )

        if "policy_regime" in clean_frame:
            regimes = clean_frame["policy_regime"].astype("string")
            if regimes.isna().any() or not regimes.isin(POLICY_REGIMES).all():
                raise ValueError("policy_rate contains invalid policy_regime values")
            clean_frame["policy_regime"] = regimes

        prepared[input_name] = clean_frame.sort_values("date").reset_index(drop=True)

    monthly_policy = (
        prepared["policy_rate"]
        .set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )
    monthly_policy[
        ["target_lower", "target_upper", "policy_rate", "policy_regime"]
    ] = monthly_policy[
        ["target_lower", "target_upper", "policy_rate", "policy_regime"]
    ].ffill()
    monthly_pce = (
        prepared["pce_yoy"]
        .set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )
    monthly_labour = (
        prepared["labour"]
        .set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )

    first_date = min(
        monthly_policy["date"].min(),
        monthly_pce["date"].min(),
        monthly_labour["date"].min(),
    )
    last_date = max(
        monthly_policy["date"].max(),
        monthly_pce["date"].max(),
        monthly_labour["date"].max(),
    )
    panel = pd.DataFrame({"date": pd.date_range(first_date, last_date, freq="ME")})
    panel = panel.merge(monthly_policy, on="date", how="left", validate="one_to_one")
    panel = panel.merge(monthly_pce, on="date", how="left", validate="one_to_one")
    panel = panel.merge(monthly_labour, on="date", how="left", validate="one_to_one")

    # Reapply the economically valid policy-state fill after the outer calendar
    # merge. This does not backfill months before official target history begins.
    panel[["target_lower", "target_upper", "policy_rate", "policy_regime"]] = (
        panel[
            ["target_lower", "target_upper", "policy_rate", "policy_regime"]
        ].ffill()
    )

    return panel.loc[
        :,
        [
            "date",
            "target_lower",
            "target_upper",
            "policy_rate",
            "policy_regime",
            "pce_index",
            "pce_yoy",
            "unemployment",
            "natural_unemployment",
        ],
    ]


def align_macro_to_meetings(
    monthly_panel: "pd.DataFrame",
    policy_rate: "pd.DataFrame",
    meetings: "pd.DataFrame",
) -> "pd.DataFrame":
    """Create one row per decision using only data available beforehand.

    Macro series are aligned independently to their latest month-end reference
    date on or before each meeting. Daily policy states are aligned strictly
    before and strictly after the decision date. This handles legacy point-target
    changes recorded on the meeting date and range changes recorded afterward.

    This is a reference-period baseline, not a fully point-in-time vintage panel:
    FRED observation dates do not say when PCE, unemployment, or NROU values were
    released or revised. The returned reference-date columns make that limitation
    auditable; production work must replace them with actual release/vintage data.
    """
    if not isinstance(monthly_panel, pd.DataFrame):
        raise TypeError("monthly_panel must be a pandas DataFrame")
    if not isinstance(policy_rate, pd.DataFrame):
        raise TypeError("policy_rate must be a pandas DataFrame")
    if not isinstance(meetings, pd.DataFrame):
        raise TypeError("meetings must be a pandas DataFrame")

    monthly_columns = {
        "date",
        "target_lower",
        "target_upper",
        "policy_rate",
        "policy_regime",
        "pce_index",
        "pce_yoy",
        "unemployment",
        "natural_unemployment",
    }
    policy_columns = {
        "date",
        "target_lower",
        "target_upper",
        "policy_rate",
        "policy_regime",
    }
    meeting_columns = {"meeting_date", "is_scheduled", "source_url"}

    for input_name, frame, required_columns in (
        ("monthly_panel", monthly_panel, monthly_columns),
        ("policy_rate", policy_rate, policy_columns),
        ("meetings", meetings, meeting_columns),
    ):
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{input_name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{input_name} contains no observations")

    monthly = monthly_panel.loc[:, sorted(monthly_columns)].copy()
    policy = policy_rate.loc[
        :,
        ["date", "target_lower", "target_upper", "policy_rate", "policy_regime"],
    ].copy()
    aligned = meetings.copy()

    monthly["date"] = pd.to_datetime(
        monthly["date"], errors="coerce"
    ).astype("datetime64[ns]")
    policy["date"] = pd.to_datetime(
        policy["date"], errors="coerce"
    ).astype("datetime64[ns]")
    aligned["meeting_date"] = pd.to_datetime(
        aligned["meeting_date"], errors="coerce"
    ).astype("datetime64[ns]")
    for input_name, frame, date_column in (
        ("monthly_panel", monthly, "date"),
        ("policy_rate", policy, "date"),
        ("meetings", aligned, "meeting_date"),
    ):
        if frame[date_column].isna().any():
            raise ValueError(f"{input_name} contains missing or invalid dates")
        if frame[date_column].duplicated().any():
            raise ValueError(f"{input_name} contains duplicate dates")

    for value_column in (
        "target_lower",
        "target_upper",
        "policy_rate",
    ):
        raw_values = policy[value_column]
        policy[value_column] = pd.to_numeric(raw_values, errors="coerce")
        if policy[value_column].isna().any():
            raise ValueError(f"policy_rate contains missing or invalid {value_column}")

    if (policy["target_lower"] > policy["target_upper"]).any():
        raise ValueError("policy_rate contains a lower bound above its upper bound")
    expected_midpoint = (policy["target_lower"] + policy["target_upper"]) / 2.0
    if ((policy["policy_rate"] - expected_midpoint).abs() > 1e-10).any():
        raise ValueError("policy_rate midpoint is inconsistent with its bounds")
    policy["policy_regime"] = policy["policy_regime"].astype("string")
    if (
        policy["policy_regime"].isna().any()
        or not policy["policy_regime"].isin(POLICY_REGIMES).all()
    ):
        raise ValueError("policy_rate contains invalid policy_regime values")

    for value_column in (
        "pce_index",
        "pce_yoy",
        "unemployment",
        "natural_unemployment",
    ):
        raw_values = monthly[value_column]
        monthly[value_column] = pd.to_numeric(raw_values, errors="coerce")
        nonempty_values = (
            raw_values.notna() & raw_values.astype("string").str.strip().ne("")
        )
        if (nonempty_values & monthly[value_column].isna()).any():
            raise ValueError(
                f"monthly_panel.{value_column} contains non-numeric values"
            )

    if aligned["source_url"].isna().any():
        raise ValueError("meetings contains missing source_url values")

    monthly = monthly.sort_values("date").reset_index(drop=True)
    policy = policy.sort_values("date").reset_index(drop=True)
    aligned = aligned.sort_values("meeting_date").reset_index(drop=True)

    def align_latest_macro(
        frame: pd.DataFrame,
        *,
        value_columns: list[str],
        required_value: str,
        reference_column: str,
    ) -> pd.DataFrame:
        available = monthly.loc[
            monthly[required_value].notna(), ["date", *value_columns]
        ].copy()
        if available.empty:
            raise ValueError(
                f"monthly_panel contains no available {required_value} observations"
            )
        available = available.rename(columns={"date": reference_column})
        return pd.merge_asof(
            frame.sort_values("meeting_date"),
            available.sort_values(reference_column),
            left_on="meeting_date",
            right_on=reference_column,
            direction="backward",
            allow_exact_matches=True,
        )

    aligned = align_latest_macro(
        aligned,
        value_columns=["pce_index", "pce_yoy"],
        required_value="pce_yoy",
        reference_column="pce_reference_date",
    )
    aligned = align_latest_macro(
        aligned,
        value_columns=["unemployment"],
        required_value="unemployment",
        reference_column="unemployment_reference_date",
    )
    aligned = align_latest_macro(
        aligned,
        value_columns=["natural_unemployment"],
        required_value="natural_unemployment",
        reference_column="natural_unemployment_reference_date",
    )

    policy_before = policy.rename(
        columns={
            "date": "policy_date_before",
            "target_lower": "target_lower_before",
            "target_upper": "target_upper_before",
            "policy_rate": "policy_rate_before",
            "policy_regime": "policy_regime_before",
        }
    )
    aligned["_before_lookup_date"] = aligned["meeting_date"] - pd.Timedelta(
        nanoseconds=1
    )
    aligned = pd.merge_asof(
        aligned.sort_values("_before_lookup_date"),
        policy_before.sort_values("policy_date_before"),
        left_on="_before_lookup_date",
        right_on="policy_date_before",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )

    policy_after = policy.rename(
        columns={
            "date": "policy_date_after",
            "target_lower": "target_lower_after",
            "target_upper": "target_upper_after",
            "policy_rate": "policy_rate_after",
            "policy_regime": "policy_regime_after",
        }
    )
    aligned["_after_lookup_date"] = aligned["meeting_date"] + pd.Timedelta(
        nanoseconds=1
    )
    aligned = pd.merge_asof(
        aligned.sort_values("_after_lookup_date"),
        policy_after.sort_values("policy_date_after"),
        left_on="_after_lookup_date",
        right_on="policy_date_after",
        direction="forward",
        tolerance=pd.Timedelta(days=7),
    )

    aligned = aligned.drop(columns=["_before_lookup_date", "_after_lookup_date"])
    aligned = aligned.dropna(
        subset=[
            "policy_date_before",
            "policy_rate_before",
            "policy_regime_before",
            "policy_date_after",
            "policy_rate_after",
            "policy_regime_after",
        ]
    )
    if aligned.empty:
        raise ValueError(
            "No meetings have both pre- and post-decision policy observations"
        )

    for reference_column in (
        "pce_reference_date",
        "unemployment_reference_date",
        "natural_unemployment_reference_date",
        "policy_date_before",
    ):
        invalid_reference = aligned[reference_column] > aligned["meeting_date"]
        if invalid_reference.any():
            raise RuntimeError(
                f"{reference_column} contains dates after their meeting"
            )
    if (aligned["policy_date_after"] <= aligned["meeting_date"]).any():
        raise RuntimeError("policy_date_after must be later than the meeting date")

    return aligned.sort_values("meeting_date").reset_index(drop=True)


def label_fomc_decisions(meeting_panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add three-class ``decision`` and binary ``is_change`` targets.

    Compare ``policy_rate_after`` with ``policy_rate_before``. Apply
    ``config.RATE_CHANGE_TOLERANCE_BPS`` after converting percentage-point
    changes to basis points: positive is ``hike``, negative is ``cut``, and a
    change within tolerance is ``hold``. Keep the basis-point change for audit.

    Only these target columns may depend on the post-meeting policy rate.
    """
    if not isinstance(meeting_panel, pd.DataFrame):
        raise TypeError("meeting_panel must be a pandas DataFrame")

    required_columns = {
        "meeting_date",
        "policy_rate_before",
        "policy_rate_after",
    }
    missing_columns = sorted(required_columns - set(meeting_panel.columns))
    if missing_columns:
        raise ValueError(
            f"meeting_panel is missing required columns: {missing_columns}"
        )
    if meeting_panel.empty:
        raise ValueError("meeting_panel contains no meetings")

    labelled = meeting_panel.copy()
    labelled["meeting_date"] = pd.to_datetime(
        labelled["meeting_date"], errors="coerce"
    )
    if labelled["meeting_date"].isna().any():
        raise ValueError("meeting_panel contains missing or invalid meeting dates")
    if labelled["meeting_date"].duplicated().any():
        raise ValueError("meeting_panel contains duplicate meeting dates")

    for rate_column in ("policy_rate_before", "policy_rate_after"):
        raw_values = labelled[rate_column]
        labelled[rate_column] = pd.to_numeric(raw_values, errors="coerce")
        if labelled[rate_column].isna().any():
            raise ValueError(f"meeting_panel contains missing or invalid {rate_column}")
        if ((labelled[rate_column] < 0) | (labelled[rate_column] > 100)).any():
            raise ValueError(f"{rate_column} values must be between 0 and 100")

    labelled["rate_change_bps"] = (
        labelled["policy_rate_after"] - labelled["policy_rate_before"]
    ) * 100.0
    labelled["decision"] = "hold"
    labelled.loc[
        labelled["rate_change_bps"] > RATE_CHANGE_TOLERANCE_BPS,
        "decision",
    ] = "hike"
    labelled.loc[
        labelled["rate_change_bps"] < -RATE_CHANGE_TOLERANCE_BPS,
        "decision",
    ] = "cut"
    labelled["is_change"] = labelled["decision"].ne("hold").astype("int8")

    return labelled.sort_values("meeting_date").reset_index(drop=True)


def validate_clean_panel(panel: "pd.DataFrame") -> None:
    """Fail loudly when the cleaned panel is unsafe for feature engineering.

    Check unique/sorted meeting dates, allowed target classes, binary-label
    agreement, target-rate arithmetic, required values, plausible ranges, and
    source/reference chronology. Return only after every check passes.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("Clean panel contains no meetings")

    required_columns = {
        "meeting_date",
        "is_scheduled",
        "source_url",
        "pce_reference_date",
        "pce_index",
        "pce_yoy",
        "unemployment_reference_date",
        "unemployment",
        "natural_unemployment_reference_date",
        "natural_unemployment",
        "policy_date_before",
        "target_lower_before",
        "target_upper_before",
        "policy_rate_before",
        "policy_regime_before",
        "policy_date_after",
        "target_lower_after",
        "target_upper_after",
        "policy_rate_after",
        "policy_regime_after",
        "rate_change_bps",
        "decision",
        "is_change",
    }
    missing_columns = sorted(required_columns - set(panel.columns))
    if missing_columns:
        raise ValueError(f"Clean panel is missing required columns: {missing_columns}")

    validated = panel.copy()
    date_columns = (
        "meeting_date",
        "pce_reference_date",
        "unemployment_reference_date",
        "natural_unemployment_reference_date",
        "policy_date_before",
        "policy_date_after",
    )
    for date_column in date_columns:
        validated[date_column] = pd.to_datetime(
            validated[date_column], errors="coerce"
        )
        if validated[date_column].isna().any():
            raise ValueError(f"Clean panel contains missing or invalid {date_column}")

    if validated["meeting_date"].duplicated().any():
        raise ValueError("Clean panel contains duplicate meeting dates")
    if not validated["meeting_date"].is_monotonic_increasing:
        raise ValueError("Clean panel must be sorted by meeting_date")

    if not pd.api.types.is_bool_dtype(validated["is_scheduled"]):
        raise ValueError("is_scheduled must contain boolean values")
    source_urls = validated["source_url"].astype("string")
    if source_urls.isna().any() or source_urls.str.strip().eq("").any():
        raise ValueError("source_url must be present for every meeting")

    numeric_columns = (
        "pce_index",
        "pce_yoy",
        "unemployment",
        "natural_unemployment",
        "target_lower_before",
        "target_upper_before",
        "policy_rate_before",
        "target_lower_after",
        "target_upper_after",
        "policy_rate_after",
        "rate_change_bps",
        "is_change",
    )
    for numeric_column in numeric_columns:
        values = pd.to_numeric(validated[numeric_column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"Clean panel contains missing or invalid {numeric_column}"
            )
        validated[numeric_column] = values

    if (validated["pce_index"] <= 0).any():
        raise ValueError("pce_index values must be greater than zero")
    for percentage_column in (
        "unemployment",
        "natural_unemployment",
        "target_lower_before",
        "target_upper_before",
        "policy_rate_before",
        "target_lower_after",
        "target_upper_after",
        "policy_rate_after",
    ):
        if (
            (validated[percentage_column] < 0)
            | (validated[percentage_column] > 100)
        ).any():
            raise ValueError(f"{percentage_column} values must be between 0 and 100")

    for suffix in ("before", "after"):
        lower = validated[f"target_lower_{suffix}"]
        upper = validated[f"target_upper_{suffix}"]
        midpoint = validated[f"policy_rate_{suffix}"]
        regime_column = f"policy_regime_{suffix}"
        regimes = validated[regime_column].astype("string")
        if regimes.isna().any() or not regimes.isin(POLICY_REGIMES).all():
            raise ValueError(f"Clean panel contains invalid {regime_column}")
        if (lower > upper).any():
            raise ValueError(
                f"Target lower bound exceeds upper bound in {suffix} values"
            )
        if not np.allclose(midpoint, (lower + upper) / 2.0, atol=1e-10):
            raise ValueError(f"Policy midpoint is inconsistent with {suffix} bounds")
        point_rows = regimes.eq("point_target")
        if not np.allclose(
            lower.loc[point_rows], upper.loc[point_rows], atol=1e-10
        ):
            raise ValueError(
                f"Point-target {suffix} rows must have identical synthetic bounds"
            )
        range_rows = regimes.eq("target_range")
        if (lower.loc[range_rows] >= upper.loc[range_rows]).any():
            raise ValueError(
                f"Target-range {suffix} rows must have lower bounds below upper bounds"
            )

    expected_bps = (
        validated["policy_rate_after"] - validated["policy_rate_before"]
    ) * 100.0
    if not np.allclose(validated["rate_change_bps"], expected_bps, atol=1e-8):
        raise ValueError("rate_change_bps is inconsistent with pre/post policy rates")

    decisions = validated["decision"].astype("string")
    invalid_decisions = ~decisions.isin(DECISION_CLASSES)
    if invalid_decisions.any():
        invalid_values = sorted(decisions.loc[invalid_decisions].unique().tolist())
        raise ValueError(f"Clean panel contains invalid decisions: {invalid_values}")

    expected_decisions = pd.Series("hold", index=validated.index, dtype="string")
    expected_decisions.loc[
        validated["rate_change_bps"] > RATE_CHANGE_TOLERANCE_BPS
    ] = "hike"
    expected_decisions.loc[
        validated["rate_change_bps"] < -RATE_CHANGE_TOLERANCE_BPS
    ] = "cut"
    if not decisions.equals(expected_decisions):
        raise ValueError("decision labels disagree with rate_change_bps")

    if not validated["is_change"].isin([0, 1]).all():
        raise ValueError("is_change must contain only 0 and 1")
    expected_is_change = decisions.ne("hold").astype("int8")
    if not validated["is_change"].astype("int8").equals(expected_is_change):
        raise ValueError("is_change disagrees with decision")

    for reference_column in (
        "pce_reference_date",
        "unemployment_reference_date",
        "natural_unemployment_reference_date",
    ):
        if (validated[reference_column] > validated["meeting_date"]).any():
            raise ValueError(f"{reference_column} cannot be after meeting_date")
    if (validated["policy_date_before"] >= validated["meeting_date"]).any():
        raise ValueError("policy_date_before must be strictly before meeting_date")
    if (validated["policy_date_after"] <= validated["meeting_date"]).any():
        raise ValueError("policy_date_after must be strictly after meeting_date")
    if (
        validated["meeting_date"] - validated["policy_date_before"]
        > pd.Timedelta(days=7)
    ).any():
        raise ValueError("A pre-meeting policy observation is more than 7 days old")
    if (
        validated["policy_date_after"] - validated["meeting_date"]
        > pd.Timedelta(days=7)
    ).any():
        raise ValueError("A post-meeting policy observation is more than 7 days late")


def build_clean_panel() -> Path:
    """Build and save the panel using the paths declared in ``config.py``."""
    raw_dir = DATA_RAW
    output_path = CLEAN_PANEL_PATH

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    target_lower = load_raw_series(raw_dir / "target_lower.csv", "target_lower")
    target_upper = load_raw_series(raw_dir / "target_upper.csv", "target_upper")
    target_rate = load_raw_series(raw_dir / "target_rate.csv", "target_rate")
    pce_index = load_raw_series(raw_dir / "pce_index.csv", "pce_index")
    unemployment = load_raw_series(raw_dir / "unemployment.csv", "unemployment")
    natural_unemployment = load_raw_series(
        raw_dir / "natural_unemployment.csv",
        "natural_unemployment",
    )

    meetings_path = raw_dir / "fomc_meetings.csv"
    if not meetings_path.is_file():
        raise FileNotFoundError(
            f"Raw FOMC calendar file does not exist: {meetings_path}"
        )
    meetings = pd.read_csv(meetings_path)

    policy_rate = build_policy_rate_series(target_rate, target_lower, target_upper)
    pce_yoy = calculate_pce_yoy(pce_index)
    labour = prepare_monthly_labour_data(unemployment, natural_unemployment)
    monthly_panel = build_monthly_macro_panel(policy_rate, pce_yoy, labour)
    meeting_panel = align_macro_to_meetings(monthly_panel, policy_rate, meetings)
    clean_panel = label_fomc_decisions(meeting_panel)

    validate_clean_panel(clean_panel)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_panel.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output_path


def main() -> None:
    """Build the clean panel using only the configured project paths."""
    saved_path = build_clean_panel()
    saved_panel = pd.read_csv(saved_path)

    print(f"Saved clean panel to {saved_path}")
    print(f"Rows: {len(saved_panel)}")
    decision_counts = saved_panel["decision"].value_counts()
    counts_text = ", ".join(
        f"{decision}={int(decision_counts.get(decision, 0))}"
        for decision in DECISION_CLASSES
    )
    print(f"Decision counts: {counts_text}")


if __name__ == "__main__":
    main()
