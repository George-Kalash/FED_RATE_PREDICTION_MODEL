"""Stage 1a: acquire structured FRED observations and the FOMC calendar."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from config import (
    DATA_RAW,
    FOMC_CALENDAR_URL,
    FOMC_MEETINGS_PATH,
    FRED_API_BASE_URL,
    FRED_API_KEY,
    FRED_API_KEY_ENV,
    FRED_SERIES,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)

def resolve_fred_api_key(cli_api_key: str | None = None) -> str:
    """Return a FRED API key without committing it to source control."""

    if cli_api_key and cli_api_key.strip():
        return cli_api_key.strip()
    if FRED_API_KEY and FRED_API_KEY.strip():
        return FRED_API_KEY.strip()
    raise RuntimeError(
        f"Missing FRED API key. Add {FRED_API_KEY_ENV}=... to contents/.env "
        "or provide it via --api-key."
    )


def fetch_fred_series(
    series_id: str,
    api_key: str,
    *,
    observation_start: str | None = None,
) -> "pd.DataFrame":
    """Fetch one FRED series and return exactly ``date`` and ``value`` columns.
    Description: 
    - Send a GET request to ``config.FRED_API_BASE_URL`` with ``file_type=json``.
    - Include the optional start date only when provided.
    - enforce the configured timeout, identify this project with its user agent,
      and raise useful HTTP/JSON errors.
    - Convert FRED's ``'.'`` missing-value marker to a real missing value.
    - Parse dates and numeric values; sort ascending; reject duplicate dates.
    - Return data only. Saving belongs in ``pull_all_model_series``.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start is not None:
        params["observation_start"] = observation_start

    response = requests.get(
        FRED_API_BASE_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(
            f"FRED request failed for {series_id} "
            f"with status {response.status_code}."
        ) from None

    payload = response.json()
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise RuntimeError(
            f"FRED response for {series_id} did not contain an observations list."
        )

    frame = pd.DataFrame(observations)
    if not {"date", "value"}.issubset(frame.columns):
        raise RuntimeError(
            f"FRED response for {series_id} did not contain date and value fields."
        )

    frame = frame.loc[:, ["date", "value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame["date"].duplicated().any():
        raise RuntimeError(f"FRED returned duplicate dates for {series_id}.")

    return frame


def pull_all_model_series(
    api_key: str,
    *,
    observation_start: str | None = None,
) -> dict[str, Path]:
    """Fetch all declared FRED series into ``config.DATA_RAW``."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for logical_name, specification in FRED_SERIES.items():
        frame = fetch_fred_series(
            series_id=specification["series_id"],
            api_key=api_key,
            observation_start=observation_start,
        )
        output_path = DATA_RAW / f"{logical_name}.csv"
        frame.to_csv(output_path, index=False)
        result[logical_name] = output_path

    return result
    # raise NotImplementedError("TODO: pull all structured model series")


def fetch_fomc_meeting_calendar(
    calendar_url: str,
    *,
    start_year: int,
    end_year: int,
) -> "pd.DataFrame":
    """Return scheduled FOMC decision dates in a ``meeting_date`` column.
    Description: 
    - Parse official meeting dates from the Federal Reserve calendar.
    - For two-day meetings, store the second day (the decision date).
    - Record unscheduled/intermeeting decisions when the historical source
      identifies them, with a boolean ``is_scheduled`` column.
    - Store a ``source_url`` column so every label date remains auditable.
    - Normalize dates, sort them, remove duplicates, and restrict the year range.
    - Add a saved fixture or parser test before trusting changing page markup.

    This calendar is label/alignment data, not a scraped predictive feature.
    """
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    month_numbers = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    print(f"Fetching FOMC meeting calendar from {calendar_url} for {start_year}-{end_year}")

    def parse_decision_date(
        year: int, month_text: str, date_text: str
    ) -> pd.Timestamp:
        """Convert a Fed display range to its final policy-decision date."""
        month_words = re.findall(r"[A-Za-z]+", month_text)
        date_words = re.findall(r"[A-Za-z]+", date_text)
        day_numbers = re.findall(r"\d{1,2}", date_text)

        if not month_words or not day_numbers:
            raise ValueError(
                f"Cannot parse FOMC date from {month_text!r} {date_text!r}"
            )

        first_month_name = month_words[0].lower()
        final_month_name = (
            date_words[-1].lower() if date_words else month_words[-1].lower()
        )
        if first_month_name not in month_numbers or final_month_name not in month_numbers:
            raise ValueError(
                f"Unknown month in FOMC date {month_text!r} {date_text!r}"
            )

        first_month = month_numbers[first_month_name]
        final_month = month_numbers[final_month_name]
        decision_year = year + 1 if first_month == 12 and final_month == 1 else year

        return pd.Timestamp(
            year=decision_year,
            month=final_month,
            day=int(day_numbers[-1]),
        )

    def historical_section_has_statement(heading: Tag) -> bool:
        """Identify conference calls that published a policy statement."""
        for element in heading.next_elements:
            if not isinstance(element, Tag):
                continue
            if element is not heading and element.name == "h5":
                break
            if (
                element.name == "a"
                and element.get_text(" ", strip=True).lower() == "statement"
            ):
                return True
        return False

    records: list[dict[str, object]] = []

    response = requests.get(
        calendar_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    # The recent calendar uses one panel per year and one .fomc-meeting row per
    # event. Its h4 text is, for example, "2026 FOMC Meetings", not just "2026".
    recent_years: set[int] = set()
    for panel in soup.select("div.panel.panel-default"):
        heading = panel.find("h4")
        if heading is None:
            continue

        year_match = re.search(
            r"\b(\d{4})\s+FOMC\s+Meetings\b",
            heading.get_text(" ", strip=True),
            flags=re.IGNORECASE,
        )
        if year_match is None:
            continue

        year = int(year_match.group(1))
        recent_years.add(year)
        if not start_year <= year <= end_year:
            continue

        for meeting in panel.select("div.fomc-meeting"):
            month_node = meeting.select_one(".fomc-meeting__month")
            date_node = meeting.select_one(".fomc-meeting__date")
            if month_node is None or date_node is None:
                continue

            row_text = meeting.get_text(" ", strip=True)
            if "notation vote" in row_text.lower():
                continue

            month_text = month_node.get_text(" ", strip=True)
            date_text = re.sub(
                r"\([^)]*\)|\*", "", date_node.get_text(" ", strip=True)
            ).strip()
            records.append(
                {
                    "meeting_date": parse_decision_date(year, month_text, date_text),
                    "is_scheduled": "unscheduled" not in row_text.lower(),
                    "source_url": calendar_url,
                }
            )

    # Older years live on one official historical page per year and use h5
    # headings such as "January 29-30 Meeting - 2008".
    historical_template = urljoin(
        calendar_url, "/monetarypolicy/fomchistorical{year}.htm"
    )
    heading_pattern = re.compile(
        r"^(?P<date>.+?)\s+(?P<kind>Meeting|Conference Call)\s+-\s+"
        r"(?P<year>\d{4})$",
        flags=re.IGNORECASE,
    )

    for year in range(start_year, end_year + 1):
        if year in recent_years:
            continue

        historical_url = historical_template.format(year=year)
        historical_response = requests.get(
            historical_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        historical_response.raise_for_status()
        historical_soup = BeautifulSoup(historical_response.content, "lxml")

        for heading in historical_soup.select("h5"):
            heading_text = " ".join(heading.get_text(" ", strip=True).split())
            match = heading_pattern.fullmatch(heading_text)
            if match is None or int(match.group("year")) != year:
                continue

            kind = match.group("kind").lower()
            if kind == "conference call" and not historical_section_has_statement(heading):
                continue

            raw_date = match.group("date")
            is_scheduled = kind == "meeting" and "unscheduled" not in raw_date.lower()
            clean_date = re.sub(r"\([^)]*\)", "", raw_date).strip()
            date_match = re.fullmatch(
                r"(?P<month>[A-Za-z]+(?:/[A-Za-z]+)?)\s+(?P<days>.+)",
                clean_date,
            )
            if date_match is None:
                raise RuntimeError(
                    f"Cannot parse historical FOMC heading {heading_text!r}"
                )

            records.append(
                {
                    "meeting_date": parse_decision_date(
                        year,
                        date_match.group("month"),
                        date_match.group("days"),
                    ),
                    "is_scheduled": is_scheduled,
                    "source_url": historical_url,
                }
            )

    calendar = pd.DataFrame.from_records(
        records,
        columns=["meeting_date", "is_scheduled", "source_url"],
    )
    if calendar.empty:
        raise RuntimeError(
            f"No FOMC meetings found for {start_year} through {end_year}; "
            "the Federal Reserve page structure may have changed."
        )

    calendar["meeting_date"] = pd.to_datetime(calendar["meeting_date"])
    calendar = calendar.loc[
        calendar["meeting_date"].dt.year.between(start_year, end_year)
    ]
    calendar = calendar.sort_values("meeting_date").reset_index(drop=True)

    duplicate_dates = calendar.loc[
        calendar["meeting_date"].duplicated(keep=False), "meeting_date"
    ]
    if not duplicate_dates.empty:
        duplicate_text = ", ".join(
            date.strftime("%Y-%m-%d") for date in duplicate_dates
        )
        raise RuntimeError(f"Duplicate FOMC meeting dates found: {duplicate_text}")

    return calendar


def save_fomc_meeting_calendar(calendar: "pd.DataFrame") -> Path:
    """Validate and save the calendar to ``config.FOMC_MEETINGS_PATH``."""
    
    
    for _, meeting in calendar.iterrows():
        if meeting["meeting_date"] is None: 
            raise ValueError("meeting date not found")
        if not isinstance(meeting["is_scheduled"], bool):
            raise ValueError("is_scheduled not found")
        if meeting["source_url"] is None:
            raise ValueError("source_url not found")
    
    FOMC_MEETINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(FOMC_MEETINGS_PATH, index=False, date_format="%Y-%m-%d")
    return FOMC_MEETINGS_PATH


def main() -> None:
    """CLI declaration for structured acquisition."""
    parser = argparse.ArgumentParser(
        description="Download model inputs for the Fed decision predictor."
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--observation-start", default=None)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year cannot be earlier than --start-year")
        
    if args.observation_start is not None:
        try: 
            date.fromisoformat(args.observation_start)
        except ValueError:
            parser.error("--observation-start must be in YYYY-MM-DD format")

    api_key = resolve_fred_api_key(args.api_key)
    series_paths = pull_all_model_series(
        api_key=api_key,
        observation_start=args.observation_start,
    )
    calendar = fetch_fomc_meeting_calendar(
        calendar_url=FOMC_CALENDAR_URL,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    calendar_path = save_fomc_meeting_calendar(calendar)
    print(f"Saved FOMC meeting calendar to {calendar_path} with {len(calendar)} rows.")
    for logical_name, path in series_paths.items():
        row_count = len(pd.read_csv(path))
        print(f"Saved {logical_name}: {path} ({row_count} rows)")
if __name__ == "__main__":
    main()
