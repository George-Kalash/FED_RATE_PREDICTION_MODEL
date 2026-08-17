"""Stage 1b: scrape coverage metadata that is not used by the model.

The output of this module documents source coverage. It must stay separate from
the structured FRED series and FOMC calendar consumed by ``clean.py``.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from config import (
    REQUEST_TIMEOUT_SECONDS,
    SCRAPE_DELAY_SECONDS,
    SCRAPE_MAX_CELL_CHARS,
    SCRAPE_MAX_HEADING_CHARS,
    SCRAPE_MAX_HEADINGS,
    SCRAPE_MAX_TABLE_COLUMNS,
    SCRAPE_MAX_TABLE_ROWS,
    SCRAPE_MAX_TABLES,
    SCRAPE_MAX_TITLE_CHARS,
    SOURCE_SCRAPE_JSONL_PATH,
    SOURCE_SCRAPE_SUMMARY_PATH,
    SUPPLEMENTARY_SOURCES,
    USER_AGENT,
    WebSourceSpec,
)


class SourceScrapeRecord(TypedDict):
    """Stable schema for one coverage-only scrape result."""

    source_id: str
    category: str
    url: str
    fetched_at_utc: str
    status_code: int | None
    title: str | None
    headings: list[str]
    table_previews: list[list[dict[str, str | None]]]
    error: str | None


def _normalize_text(value: str, character_limit: int) -> str:
    """Collapse whitespace and limit one extracted text field."""
    print(f"Normalizing text: {value!r} with limit {character_limit}")
    normalized = " ".join(value.split())
    if len(normalized) <= character_limit:
        return normalized
    return normalized[: character_limit - 1].rstrip() + "…"


def _validate_source(source: WebSourceSpec) -> tuple[str, str, str]:
    """Validate a source declaration and return normalized fields."""
    print(f"Validating source: {source}")
    required_fields = {"id", "category", "url"}
    if not isinstance(source, dict) or not required_fields.issubset(source):
        raise ValueError(f"source must contain {sorted(required_fields)}")

    values: dict[str, str] = {}
    for field in required_fields:
        raw_value = source[field]
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"source {field} must be a non-empty string")
        values[field] = raw_value.strip()

    parsed_url = urlsplit(values["url"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"source {values['id']!r} has an invalid HTTP(S) URL")
    return values["id"], values["category"], values["url"]


def _table_preview(table: Tag) -> list[dict[str, str | None]]:
    """Convert a small part of one HTML table to JSON-compatible rows."""
    print(f"Extracting table preview from: {table}")
    html_rows = table.find_all("tr", limit=SCRAPE_MAX_TABLE_ROWS + 1)
    if not html_rows:
        return []

    first_cells = html_rows[0].find_all(["th", "td"], recursive=False)
    has_header = any(cell.name == "th" for cell in first_cells)
    data_rows = html_rows[1:] if has_header else html_rows[:SCRAPE_MAX_TABLE_ROWS]
    data_rows = data_rows[:SCRAPE_MAX_TABLE_ROWS]

    row_cells = [
        row.find_all(["th", "td"], recursive=False) for row in data_rows
    ]
    widest_row = max(
        [len(first_cells) if has_header else 0, *(len(cells) for cells in row_cells)],
        default=0,
    )
    column_count = min(widest_row, SCRAPE_MAX_TABLE_COLUMNS)
    if column_count == 0:
        return []

    raw_headers = []
    if has_header:
        raw_headers = [
            _normalize_text(cell.get_text(" ", strip=True), SCRAPE_MAX_HEADING_CHARS)
            for cell in first_cells[:column_count]
        ]

    headers: list[str] = []
    occurrences: dict[str, int] = {}
    for position in range(column_count):
        base = (
            raw_headers[position]
            if position < len(raw_headers) and raw_headers[position]
            else f"column_{position + 1}"
        )
        occurrences[base] = occurrences.get(base, 0) + 1
        suffix = occurrences[base]
        headers.append(base if suffix == 1 else f"{base}_{suffix}")

    preview: list[dict[str, str | None]] = []
    for cells in row_cells:
        record: dict[str, str | None] = {}
        for position, header in enumerate(headers):
            if position >= len(cells):
                record[header] = None
                continue
            value = _normalize_text(
                cells[position].get_text(" ", strip=True),
                SCRAPE_MAX_CELL_CHARS,
            )
            record[header] = value or None
        preview.append(record)
    return preview


def _robots_permission(url: str) -> tuple[bool, str | None]:
    """Check the origin's robots.txt using the configured user agent."""
    print(f"Checking robots.txt permission for: {url}")
    parts = urlsplit(url)
    robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))

    try:
        response = requests.get(
            robots_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException as error:
        return False, f"Could not verify robots.txt: {error}"

    if response.status_code in {401, 403}:
        return False, f"robots.txt access returned HTTP {response.status_code}"
    if 400 <= response.status_code < 500:
        return True, None
    try:
        response.raise_for_status()
    except requests.RequestException as error:
        return False, f"Could not verify robots.txt: {error}"

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())
    if not parser.can_fetch(USER_AGENT, url):
        return False, "robots.txt disallows this URL for the configured user agent"
    return True, None


def scrape_page(source: WebSourceSpec) -> SourceScrapeRecord:
    print(f"Scraping page for source: {source}")
    """Extract descriptive metadata from one supplementary page.

    Do not turn scraped text or tables into model features in this function.
    """
    source_id, category, url = _validate_source(source)

    record: SourceScrapeRecord = {
        "source_id": source_id,
        "category": category,
        "url": url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status_code": None,
        "title": None,
        "headings": [],
        "table_previews": [],
        "error": None,
    }

    allowed, robots_error = _robots_permission(url)
    if not allowed:
        record["error"] = robots_error
        return record

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        record["status_code"] = response.status_code
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            raise ValueError(f"Expected HTML but received {content_type!r}")

        soup = BeautifulSoup(response.content, "lxml")
        if soup.title is not None:
            title = _normalize_text(
                soup.title.get_text(" ", strip=True),
                SCRAPE_MAX_TITLE_CHARS,
            )
            record["title"] = title or None

        seen_headings: set[str] = set()
        for heading in soup.select("h1, h2, h3"):
            text = _normalize_text(
                heading.get_text(" ", strip=True),
                SCRAPE_MAX_HEADING_CHARS,
            )
            if not text or text in seen_headings:
                continue
            seen_headings.add(text)
            record["headings"].append(text)
            if len(record["headings"]) == SCRAPE_MAX_HEADINGS:
                break

        for table in soup.find_all("table", limit=SCRAPE_MAX_TABLES):
            preview = _table_preview(table)
            if preview:
                record["table_previews"].append(preview)
    except (requests.RequestException, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"

    return record


def scrape_all_sources() -> list[SourceScrapeRecord]:
    """Scrape configured pages while preserving one result per source.

    Source declarations and pacing come from ``config.py``. Failed page fetches
    remain in the returned list as records with a populated ``error`` field.
    """
    print("Starting scrape of all supplementary sources...")
    seen_ids: set[str] = set()
    for source in SUPPLEMENTARY_SOURCES:
        source_id, _, _ = _validate_source(source)
        if source_id in seen_ids:
            raise ValueError(f"Duplicate supplementary source id: {source_id}")
        seen_ids.add(source_id)

    records: list[SourceScrapeRecord] = []
    for position, source in enumerate(SUPPLEMENTARY_SOURCES):
        records.append(scrape_page(source))
        if position < len(SUPPLEMENTARY_SOURCES) - 1:
            time.sleep(SCRAPE_DELAY_SECONDS)
    return records


def _validate_records(records: list[SourceScrapeRecord]) -> None:
    """Validate the shared scrape-record contract before writing artifacts."""
    print(f"Validating {len(records)} scrape records...")
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    if not records:
        raise ValueError("records must contain at least one scrape result")

    required_fields = set(SourceScrapeRecord.__required_keys__)
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {position} must be a dictionary")
        missing = sorted(required_fields - set(record))
        if missing:
            raise ValueError(f"Record {position} is missing fields: {missing}")
        unexpected = sorted(set(record) - required_fields)
        if unexpected:
            raise ValueError(f"Record {position} has unexpected fields: {unexpected}")

        for text_field in ("source_id", "category", "url", "fetched_at_utc"):
            if not isinstance(record[text_field], str) or not record[text_field].strip():
                raise ValueError(
                    f"Record {position} field {text_field!r} must be non-empty text"
                )
        if record["status_code"] is not None and (
            isinstance(record["status_code"], bool)
            or not isinstance(record["status_code"], int)
        ):
            raise ValueError(f"Record {position} has an invalid status_code")
        for optional_text_field in ("title", "error"):
            value = record[optional_text_field]
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"Record {position} field {optional_text_field!r} "
                    "must be text or null"
                )
        if not isinstance(record["headings"], list) or not all(
            isinstance(heading, str) for heading in record["headings"]
        ):
            raise ValueError(f"Record {position} has invalid headings")
        if not isinstance(record["table_previews"], list):
            raise ValueError(f"Record {position} has invalid table_previews")
        try:
            json.dumps(record, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Record {position} is not valid JSON: {error}") from error


def write_jsonl(records: list[SourceScrapeRecord]) -> Path:
    """Atomically write records to ``config.SOURCE_SCRAPE_JSONL_PATH``."""
    print(f"Writing {len(records)} scrape records to JSONL...")
    _validate_records(records)
    output_path = SOURCE_SCRAPE_JSONL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def write_summary_csv(records: list[SourceScrapeRecord]) -> Path:
    """Atomically write the configured compact coverage-summary CSV."""
    print(f"Writing {len(records)} scrape records to summary CSV...")
    _validate_records(records)
    output_path = SOURCE_SCRAPE_SUMMARY_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "category",
        "url",
        "fetched_at_utc",
        "status_code",
        "title",
        "heading_count",
        "table_count",
        "error",
    ]

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "source_id": record["source_id"],
                        "category": record["category"],
                        "url": record["url"],
                        "fetched_at_utc": record["fetched_at_utc"],
                        "status_code": record["status_code"],
                        "title": record["title"],
                        "heading_count": len(record["headings"]),
                        "table_count": len(record["table_previews"]),
                        "error": record["error"],
                    }
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def main() -> None:
    """Scrape configured coverage sources and write both audit artifacts."""
    records = scrape_all_sources()
    jsonl_path = write_jsonl(records)
    summary_path = write_summary_csv(records)

    successful = sum(record["error"] is None for record in records)
    failed_ids = [
        record["source_id"] for record in records if record["error"] is not None
    ]
    print(f"Coverage scrape complete: {successful} succeeded, {len(failed_ids)} failed")
    print(f"Saved detailed records to {jsonl_path}")
    print(f"Saved summary to {summary_path}")
    if failed_ids:
        print(f"Failed sources: {', '.join(failed_ids)}")


if __name__ == "__main__":
    main()
