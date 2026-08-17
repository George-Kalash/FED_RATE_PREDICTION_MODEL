"""Stage 3: engineer pre-decision features from the clean meeting panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clean import (
    build_policy_rate_series,
    calculate_pce_yoy,
    load_raw_series,
    validate_clean_panel,
)
from config import (
    CLEAN_PANEL_PATH,
    DATA_RAW,
    DECISION_CLASSES,
    DIAGNOSTIC_TARGET,
    FEATURE_COLUMNS,
    FEATURE_PANEL_PATH,
    HAWK_DOVE_LABOUR_WEIGHT,
    INFLATION_TARGET_PERCENT,
    PRIMARY_TARGET,
    INFLATION_AVERAGE_WINDOWS_MONTHS,
    INFLATION_CHANGE_WINDOWS_MONTHS,
    RATE_CHANGE_WINDOWS_MONTHS,
    UNEMPLOYMENT_AVERAGE_WINDOWS_MONTHS,
    UNEMPLOYMENT_CHANGE_WINDOWS_MONTHS,
)


def add_rate_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add the pre-meeting rate level and calendar-month rate changes.

    ``rate_chg_1m`` and ``rate_chg_3m`` compare the rate immediately before the
    decision with the rate in force one and three calendar months earlier. They
    are deliberately not changes over one or three prior FOMC meetings.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")

    required_columns = {"meeting_date", "policy_date_before", "policy_rate_before"}
    missing = sorted(required_columns - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing rate-feature inputs: {missing}")

    featured = panel.copy()
    for date_column in ("meeting_date", "policy_date_before"):
        featured[date_column] = pd.to_datetime(
            featured[date_column], errors="coerce"
        )
        if featured[date_column].isna().any():
            raise ValueError(f"panel contains missing or invalid {date_column}")
    if featured["meeting_date"].duplicated().any():
        raise ValueError("panel contains duplicate meeting dates")
    if (featured["policy_date_before"] >= featured["meeting_date"]).any():
        raise ValueError("policy_date_before must be strictly before meeting_date")

    featured["policy_rate_before"] = pd.to_numeric(
        featured["policy_rate_before"], errors="coerce"
    )
    if featured["policy_rate_before"].isna().any():
        raise ValueError("panel contains missing or invalid policy_rate_before")

    target_rate = load_raw_series(DATA_RAW / "target_rate.csv", "target_rate")
    target_lower = load_raw_series(DATA_RAW / "target_lower.csv", "target_lower")
    target_upper = load_raw_series(DATA_RAW / "target_upper.csv", "target_upper")
    policy_history = build_policy_rate_series(
        target_rate,
        target_lower,
        target_upper,
    ).loc[
        :, ["date", "policy_rate"]
    ]
    policy_history = policy_history.sort_values("date").reset_index(drop=True)

    audit_rows = featured.loc[:, ["policy_date_before", "policy_rate_before"]].copy()
    audit_rows["_row_id"] = np.arange(len(audit_rows))
    audit_rows = audit_rows.sort_values("policy_date_before")
    audited = pd.merge_asof(
        audit_rows,
        policy_history.rename(
            columns={"date": "policy_date_before", "policy_rate": "raw_rate"}
        ),
        on="policy_date_before",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id")
    if audited["raw_rate"].isna().any() or not np.allclose(
        audited["policy_rate_before"], audited["raw_rate"], atol=1e-10
    ):
        raise ValueError("clean policy rates disagree with configured raw target bounds")

    featured["rate_level"] = featured["policy_rate_before"]
    for months in RATE_CHANGE_WINDOWS_MONTHS:
        lookup_rows = pd.DataFrame(
            {
                "_row_id": np.arange(len(featured)),
                "lookup_date": featured["meeting_date"] - pd.DateOffset(months=months),
            }
        ).sort_values("lookup_date")
        historical_rates = pd.merge_asof(
            lookup_rows,
            policy_history.rename(
                columns={"date": "lookup_date", "policy_rate": "historical_rate"}
            ),
            on="lookup_date",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row_id")
        featured[f"rate_chg_{months}m"] = (
            featured["rate_level"].to_numpy()
            - historical_rates["historical_rate"].to_numpy()
        )

    featured.attrs["rate_change_window_unit"] = "calendar_months"
    return featured


def add_inflation_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add calendar-month PCE momentum and trailing-average features.

    Transformations are calculated on the complete monthly PCE history before
    being matched to each row's ``pce_reference_date``. Rolling averages require
    complete trailing windows and include only the reference month and earlier.
    The clean panel's documented FRED-vintage limitation still applies.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")

    required_columns = {
        "meeting_date",
        "pce_reference_date",
        "pce_index",
        "pce_yoy",
    }
    missing = sorted(required_columns - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing inflation-feature inputs: {missing}")

    featured = panel.copy()
    for date_column in ("meeting_date", "pce_reference_date"):
        featured[date_column] = pd.to_datetime(
            featured[date_column], errors="coerce"
        )
        if featured[date_column].isna().any():
            raise ValueError(f"panel contains missing or invalid {date_column}")
    if (featured["pce_reference_date"] > featured["meeting_date"]).any():
        raise ValueError("pce_reference_date cannot be after meeting_date")

    for value_column in ("pce_index", "pce_yoy"):
        featured[value_column] = pd.to_numeric(
            featured[value_column], errors="coerce"
        )
        if featured[value_column].isna().any():
            raise ValueError(f"panel contains missing or invalid {value_column}")

    pce_index = load_raw_series(DATA_RAW / "pce_index.csv", "pce_index")
    monthly = calculate_pce_yoy(pce_index).sort_values("date").reset_index(drop=True)
    for months in INFLATION_CHANGE_WINDOWS_MONTHS:
        suffix = "" if months == 1 else str(months)
        monthly[f"pce_yoy_chg{suffix}"] = monthly["pce_yoy"].diff(months)
    for months in INFLATION_AVERAGE_WINDOWS_MONTHS:
        monthly[f"pce_yoy_ma{months}"] = monthly["pce_yoy"].rolling(
            window=months,
            min_periods=months,
        ).mean()

    monthly = monthly.set_index("date")
    reference_dates = pd.DatetimeIndex(featured["pce_reference_date"])
    aligned = monthly.reindex(reference_dates)
    if aligned[["pce_index", "pce_yoy"]].isna().any().any():
        missing_dates = sorted(
            featured.loc[
                aligned["pce_yoy"].isna().to_numpy(), "pce_reference_date"
            ].dt.strftime("%Y-%m-%d").unique()
        )
        raise ValueError(f"Raw PCE history is missing reference dates: {missing_dates}")
    if not np.allclose(
        featured["pce_index"].to_numpy(), aligned["pce_index"].to_numpy(), atol=1e-10
    ) or not np.allclose(
        featured["pce_yoy"].to_numpy(), aligned["pce_yoy"].to_numpy(), atol=1e-10
    ):
        raise ValueError("clean PCE values disagree with configured raw PCE history")

    inflation_feature_columns = [
        *("pce_yoy_chg" if months == 1 else f"pce_yoy_chg{months}"
          for months in INFLATION_CHANGE_WINDOWS_MONTHS),
        *(f"pce_yoy_ma{months}" for months in INFLATION_AVERAGE_WINDOWS_MONTHS),
    ]
    for feature_column in inflation_feature_columns:
        featured[feature_column] = aligned[feature_column].to_numpy()

    featured.attrs["inflation_window_unit"] = "calendar_months"
    return featured


def add_labour_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add calendar-month unemployment momentum and trailing averages."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")

    required_columns = {
        "meeting_date",
        "unemployment_reference_date",
        "unemployment",
        "natural_unemployment_reference_date",
        "natural_unemployment",
    }
    missing = sorted(required_columns - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing labour-feature inputs: {missing}")

    featured = panel.copy()
    for date_column in (
        "meeting_date",
        "unemployment_reference_date",
        "natural_unemployment_reference_date",
    ):
        featured[date_column] = pd.to_datetime(
            featured[date_column], errors="coerce"
        )
        if featured[date_column].isna().any():
            raise ValueError(f"panel contains missing or invalid {date_column}")
    for reference_column in (
        "unemployment_reference_date",
        "natural_unemployment_reference_date",
    ):
        if (featured[reference_column] > featured["meeting_date"]).any():
            raise ValueError(f"{reference_column} cannot be after meeting_date")

    for value_column in ("unemployment", "natural_unemployment"):
        featured[value_column] = pd.to_numeric(
            featured[value_column], errors="coerce"
        )
        if featured[value_column].isna().any():
            raise ValueError(f"panel contains missing or invalid {value_column}")
        if (
            (featured[value_column] < 0) | (featured[value_column] > 100)
        ).any():
            raise ValueError(f"{value_column} values must be between 0 and 100")

    unemployment = load_raw_series(
        DATA_RAW / "unemployment.csv",
        "unemployment",
    )
    monthly = (
        unemployment.set_index("date")
        .resample("ME")
        .last()
        .rename_axis("date")
        .reset_index()
    )
    for months in UNEMPLOYMENT_CHANGE_WINDOWS_MONTHS:
        suffix = "" if months == 1 else str(months)
        monthly[f"unemp_chg{suffix}"] = monthly["unemployment"].diff(months)
    for months in UNEMPLOYMENT_AVERAGE_WINDOWS_MONTHS:
        monthly[f"unemp_ma{months}"] = monthly["unemployment"].rolling(
            window=months,
            min_periods=months,
        ).mean()

    monthly = monthly.set_index("date")
    reference_dates = pd.DatetimeIndex(featured["unemployment_reference_date"])
    aligned = monthly.reindex(reference_dates)
    if aligned["unemployment"].isna().any():
        missing_dates = sorted(
            featured.loc[
                aligned["unemployment"].isna().to_numpy(),
                "unemployment_reference_date",
            ].dt.strftime("%Y-%m-%d").unique()
        )
        raise ValueError(
            f"Raw unemployment history is missing reference dates: {missing_dates}"
        )
    if not np.allclose(
        featured["unemployment"].to_numpy(),
        aligned["unemployment"].to_numpy(),
        atol=1e-10,
    ):
        raise ValueError(
            "clean unemployment values disagree with configured raw history"
        )

    labour_feature_columns = [
        *("unemp_chg" if months == 1 else f"unemp_chg{months}"
          for months in UNEMPLOYMENT_CHANGE_WINDOWS_MONTHS),
        *(f"unemp_ma{months}" for months in UNEMPLOYMENT_AVERAGE_WINDOWS_MONTHS),
    ]
    for feature_column in labour_feature_columns:
        featured[feature_column] = aligned[feature_column].to_numpy()

    featured.attrs["labour_window_unit"] = "calendar_months"
    return featured


def add_policy_context_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add interpretable gap, stance, and composite features.

    Preserve ``natural_unemployment`` as a feature and do not replace it with a
    fixed Canadian neutral-unemployment assumption.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")

    required_columns = {
        "rate_level",
        "pce_yoy",
        "pce_yoy_ma3",
        "unemployment",
        "natural_unemployment",
    }
    missing = sorted(required_columns - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing policy-context inputs: {missing}")

    featured = panel.copy()
    for value_column in required_columns:
        featured[value_column] = pd.to_numeric(
            featured[value_column], errors="coerce"
        )
        nonempty_values = (
            panel[value_column].notna()
            & panel[value_column].astype("string").str.strip().ne("")
        )
        if (nonempty_values & featured[value_column].isna()).any():
            raise ValueError(f"panel contains invalid numeric {value_column}")

    featured["real_rate_proxy"] = featured["rate_level"] - featured["pce_yoy"]
    featured["inflation_gap"] = (
        featured["pce_yoy"] - INFLATION_TARGET_PERCENT
    )
    featured["labour_gap"] = (
        featured["natural_unemployment"] - featured["unemployment"]
    )
    featured["hawk_dove_score"] = (
        featured["inflation_gap"]
        + HAWK_DOVE_LABOUR_WEIGHT * featured["labour_gap"]
    )
    featured["abs_inflation_gap"] = featured["inflation_gap"].abs()
    featured["policy_tightness"] = (
        featured["rate_level"] - featured["pce_yoy_ma3"]
    )
    return featured


def _prior_nonhold_streak(decisions: "pd.Series") -> "pd.Series":
    """Return the consecutive non-hold streak known before each meeting."""
    encoded = decisions.astype("string").map({"cut": -1, "hold": 0, "hike": 1})
    if encoded.isna().any():
        raise ValueError("decision contains invalid values for streak calculation")
    result: list[int] = []
    previous_direction = 0
    streak = 0
    for current_direction in encoded.astype(int):
        result.append(streak)
        if current_direction == 0:
            previous_direction = 0
            streak = 0
        elif current_direction == previous_direction:
            streak += 1
        else:
            previous_direction = current_direction
            streak = 1
    return pd.Series(result, index=decisions.index, dtype="int64")


def add_meeting_history_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Add decision-cycle features using only meetings strictly before each row.

    Target columns are permitted here only after an explicit one-row shift. The
    first meeting receives neutral history values; the large capped value for
    ``days_since_prior_change`` means that no prior change is observed in the
    available panel rather than pretending a recent change occurred.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")
    required = {
        "meeting_date",
        "is_scheduled",
        "decision",
        "is_change",
        "rate_change_bps",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing meeting-history inputs: {missing}")

    featured = panel.copy()
    featured["meeting_date"] = pd.to_datetime(
        featured["meeting_date"], errors="coerce"
    )
    if featured["meeting_date"].isna().any():
        raise ValueError("panel contains missing or invalid meeting_date")
    if featured["meeting_date"].duplicated().any():
        raise ValueError("panel contains duplicate meeting dates")
    if not featured["meeting_date"].is_monotonic_increasing:
        raise ValueError("panel must be sorted before adding meeting history")

    decisions = featured["decision"].astype("string")
    if not decisions.isin(DECISION_CLASSES).all():
        invalid = sorted(decisions.loc[~decisions.isin(DECISION_CLASSES)].unique())
        raise ValueError(f"decision contains invalid values: {invalid}")
    is_change = pd.to_numeric(featured["is_change"], errors="coerce")
    if is_change.isna().any() or not is_change.isin([0, 1]).all():
        raise ValueError("is_change must contain only 0 and 1")
    rate_change_bps = pd.to_numeric(
        featured["rate_change_bps"], errors="coerce"
    )
    if rate_change_bps.isna().any():
        raise ValueError("rate_change_bps contains missing or invalid values")

    scheduled_text = featured["is_scheduled"].astype("string").str.lower()
    scheduled = scheduled_text.map({"true": 1, "false": 0, "1": 1, "0": 0})
    if scheduled.isna().any():
        raise ValueError("is_scheduled must contain boolean values")
    featured["is_scheduled"] = scheduled.astype("int8")

    signed = decisions.map({"cut": -1, "hold": 0, "hike": 1}).astype("int8")
    featured["prior_decision"] = signed.shift(1).fillna(0).astype("int8")
    featured["prior_is_change"] = is_change.shift(1).fillna(0).astype("int8")
    featured["prior2_is_change"] = is_change.shift(2).fillna(0).astype("int8")
    featured["prior3_change_count"] = (
        is_change.shift(1).rolling(3, min_periods=1).sum().fillna(0).astype("int8")
    )
    featured["prior3_direction"] = (
        signed.shift(1).rolling(3, min_periods=1).sum().fillna(0).astype("int8")
    )
    featured["prior_rate_change_bps"] = rate_change_bps.shift(1).fillna(0.0)
    featured["same_direction_streak"] = _prior_nonhold_streak(decisions)
    featured["days_since_prior_meeting"] = (
        featured["meeting_date"].diff().dt.days.fillna(0).astype("int64")
    )
    last_change_date = (
        featured["meeting_date"].where(is_change.eq(1)).shift(1).ffill()
    )
    featured["days_since_prior_change"] = (
        (featured["meeting_date"] - last_change_date)
        .dt.days.fillna(3650)
        .clip(lower=0, upper=3650)
        .astype("int64")
    )

    if not featured["prior_is_change"].equals(
        is_change.shift(1).fillna(0).astype("int8")
    ):
        raise RuntimeError("prior_is_change was not built with a strict lag")
    if not featured["prior_rate_change_bps"].equals(
        rate_change_bps.shift(1).fillna(0.0)
    ):
        raise RuntimeError("prior_rate_change_bps was not built with a strict lag")
    exact_history = {
        "prior_decision": signed.shift(1).fillna(0).astype("int8"),
        "prior2_is_change": is_change.shift(2).fillna(0).astype("int8"),
        "prior3_change_count": (
            is_change.shift(1).rolling(3, min_periods=1).sum().fillna(0)
        ),
        "prior3_direction": (
            signed.shift(1).rolling(3, min_periods=1).sum().fillna(0)
        ),
        "same_direction_streak": _prior_nonhold_streak(decisions),
        "days_since_prior_meeting": (
            featured["meeting_date"].diff().dt.days.fillna(0)
        ),
        "days_since_prior_change": (
            (featured["meeting_date"] - last_change_date)
            .dt.days.fillna(3650)
            .clip(lower=0, upper=3650)
        ),
    }
    for feature_column, expected_values in exact_history.items():
        if not np.allclose(
            featured[feature_column], expected_values, rtol=0, atol=1e-10
        ):
            raise RuntimeError(
                f"{feature_column} was not built exclusively from prior meetings"
            )
    return featured


def select_model_columns(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Return identifiers, configured features, and both targets.

    Incomplete feature rows are reported and removed here, after every feature
    group has been calculated. Post-meeting columns used to construct the label
    are deliberately excluded from the returned modeling panel.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel contains no meetings")

    selected_columns = [
        "meeting_date",
        *FEATURE_COLUMNS,
        PRIMARY_TARGET,
        DIAGNOSTIC_TARGET,
    ]
    missing_columns = sorted(set(selected_columns) - set(panel.columns))
    if missing_columns:
        raise ValueError(f"panel is missing model columns: {missing_columns}")

    known_label_columns = {
        "policy_date_after",
        "target_lower_after",
        "target_upper_after",
        "policy_rate_after",
        "policy_regime_after",
        "rate_change_bps",
        PRIMARY_TARGET,
        DIAGNOSTIC_TARGET,
    }
    suspicious_markers = ("future", "lead", "next_", "_after", "label")
    unknown_target_columns = sorted(
        column
        for column in panel.columns
        if column not in known_label_columns
        and any(marker in column.lower() for marker in suspicious_markers)
    )
    if unknown_target_columns:
        raise ValueError(
            "panel contains unrecognized future/target-derived columns: "
            f"{unknown_target_columns}"
        )
    configured_leakage = sorted(set(FEATURE_COLUMNS) & known_label_columns)
    if configured_leakage:
        raise ValueError(
            f"Configured features include target-derived columns: {configured_leakage}"
        )

    selected = panel.loc[:, selected_columns].copy()
    selected["meeting_date"] = pd.to_datetime(
        selected["meeting_date"], errors="coerce"
    )
    if selected["meeting_date"].isna().any():
        raise ValueError("panel contains missing or invalid meeting_date")
    if selected["meeting_date"].duplicated().any():
        raise ValueError("panel contains duplicate meeting dates")

    for feature_column in FEATURE_COLUMNS:
        raw_values = selected[feature_column]
        selected[feature_column] = pd.to_numeric(raw_values, errors="coerce")
        nonempty_values = (
            raw_values.notna() & raw_values.astype("string").str.strip().ne("")
        )
        if (nonempty_values & selected[feature_column].isna()).any():
            raise ValueError(f"Feature {feature_column} contains non-numeric values")

    if selected[[PRIMARY_TARGET, DIAGNOSTIC_TARGET]].isna().any().any():
        raise ValueError("Target columns cannot contain missing values")
    decisions = selected[DIAGNOSTIC_TARGET].astype("string")
    if not decisions.isin(DECISION_CLASSES).all():
        invalid = sorted(decisions.loc[~decisions.isin(DECISION_CLASSES)].unique())
        raise ValueError(f"decision contains invalid classes: {invalid}")

    missing_counts = selected.loc[:, FEATURE_COLUMNS].isna().sum()
    missing_counts = missing_counts.loc[missing_counts > 0]
    incomplete_rows = selected.loc[:, FEATURE_COLUMNS].isna().any(axis=1)
    if incomplete_rows.any():
        details = ", ".join(
            f"{column}={int(count)}" for column, count in missing_counts.items()
        )
        print(
            f"Dropping {int(incomplete_rows.sum())} meetings with incomplete "
            f"features ({details})"
        )
        selected = selected.loc[~incomplete_rows].copy()

    return selected.sort_values("meeting_date").reset_index(drop=True)


def validate_feature_panel(panel: "pd.DataFrame") -> None:
    """Check the final feature schema, arithmetic, targets, and chronology.

    Reference-date chronology is checked by the feature-group functions before
    those audit columns are removed by ``select_model_columns``.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("Feature panel contains no meetings")

    expected_columns = [
        "meeting_date",
        *FEATURE_COLUMNS,
        PRIMARY_TARGET,
        DIAGNOSTIC_TARGET,
    ]
    missing = sorted(set(expected_columns) - set(panel.columns))
    unexpected = sorted(set(panel.columns) - set(expected_columns))
    if missing or unexpected:
        raise ValueError(
            f"Feature panel schema mismatch; missing={missing}, unexpected={unexpected}"
        )
    if panel.columns.tolist() != expected_columns:
        raise ValueError("Feature panel columns are not in the configured order")

    validated = panel.copy()
    validated["meeting_date"] = pd.to_datetime(
        validated["meeting_date"], errors="coerce"
    )
    if validated["meeting_date"].isna().any():
        raise ValueError("Feature panel contains missing or invalid meeting dates")
    if validated["meeting_date"].duplicated().any():
        raise ValueError("Feature panel contains duplicate meeting dates")
    if not validated["meeting_date"].is_monotonic_increasing:
        raise ValueError("Feature panel must be sorted by meeting_date")

    for feature_column in FEATURE_COLUMNS:
        validated[feature_column] = pd.to_numeric(
            validated[feature_column], errors="coerce"
        )
        values = validated[feature_column].to_numpy(dtype=float)
        if np.isnan(values).any() or not np.isfinite(values).all():
            raise ValueError(f"Feature {feature_column} contains missing/non-finite values")
        if validated[feature_column].nunique(dropna=False) <= 1:
            raise ValueError(f"Feature {feature_column} has no variation")

    for percentage_column in ("rate_level", "unemployment", "natural_unemployment"):
        if (
            (validated[percentage_column] < 0)
            | (validated[percentage_column] > 100)
        ).any():
            raise ValueError(f"Feature {percentage_column} must be between 0 and 100")
    if (validated["abs_inflation_gap"] < 0).any():
        raise ValueError("abs_inflation_gap cannot be negative")

    decisions = validated[DIAGNOSTIC_TARGET].astype("string")
    if decisions.isna().any() or not decisions.isin(DECISION_CLASSES).all():
        invalid = sorted(
            decisions.loc[decisions.isna() | ~decisions.isin(DECISION_CLASSES)]
            .dropna()
            .unique()
        )
        raise ValueError(f"Feature panel contains invalid decisions: {invalid}")
    is_change = pd.to_numeric(validated[PRIMARY_TARGET], errors="coerce")
    if is_change.isna().any() or not is_change.isin([0, 1]).all():
        raise ValueError(f"{PRIMARY_TARGET} must contain only 0 and 1")
    expected_change = decisions.ne("hold").astype("int8")
    if not is_change.astype("int8").equals(expected_change):
        raise ValueError(f"{PRIMARY_TARGET} disagrees with {DIAGNOSTIC_TARGET}")

    if not validated["is_scheduled"].isin([0, 1]).all():
        raise ValueError("is_scheduled must contain only 0 and 1")
    if (validated["same_direction_streak"] < 0).any():
        raise ValueError("same_direction_streak cannot be negative")
    if (validated["days_since_prior_change"] < 0).any():
        raise ValueError("days_since_prior_change cannot be negative")

    expected_context = {
        "abs_inflation_gap": (
            validated["pce_yoy"] - INFLATION_TARGET_PERCENT
        ).abs(),
    }
    for feature_column, expected_values in expected_context.items():
        if feature_column not in FEATURE_COLUMNS:
            continue
        if not np.allclose(
            validated[feature_column], expected_values, rtol=1e-10, atol=1e-10
        ):
            raise ValueError(f"Feature {feature_column} fails its declared identity")


def build_feature_panel() -> Path:
    """Build the feature panel using only paths declared in ``config.py``."""
    if not CLEAN_PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"Clean panel does not exist: {CLEAN_PANEL_PATH}. Run clean.py first."
        )

    panel = pd.read_csv(CLEAN_PANEL_PATH)
    validate_clean_panel(panel)
    panel = add_rate_features(panel)
    panel = add_inflation_features(panel)
    panel = add_labour_features(panel)
    panel = add_policy_context_features(panel)
    panel = add_meeting_history_features(panel)
    feature_panel = select_model_columns(panel)
    validate_feature_panel(feature_panel)

    FEATURE_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_panel.to_csv(
        FEATURE_PANEL_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )
    return FEATURE_PANEL_PATH


def main() -> None:
    """Build and report the configured feature panel."""
    saved_path = build_feature_panel()
    saved_panel = pd.read_csv(saved_path)
    validate_feature_panel(saved_panel)

    print(f"Saved feature panel to {saved_path}")
    print(f"Rows: {len(saved_panel)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    decision_counts = saved_panel[DIAGNOSTIC_TARGET].value_counts()
    counts_text = ", ".join(
        f"{decision}={int(decision_counts.get(decision, 0))}"
        for decision in DECISION_CLASSES
    )
    print(f"Decision counts: {counts_text}")


if __name__ == "__main__":
    main()
