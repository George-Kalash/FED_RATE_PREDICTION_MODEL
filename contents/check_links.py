"""Optional QA declarations for supplementary source URLs.

This utility checks coverage metadata only. Link status is not a model feature
and should never determine which structured observations enter the panel.
"""

from __future__ import annotations

from typing import TypedDict

from config import WebSourceSpec


class LinkCheckResult(TypedDict):
    """Schema for one non-destructive link check."""

    source_id: str
    url: str
    is_valid_url: bool
    status_code: int | None
    is_reachable: bool
    error: str | None


def is_url_format_valid(url: str) -> bool:
    """Return whether a URL has an HTTP(S) scheme and non-empty host.

    TODO: parse with ``urllib.parse.urlparse`` and explicitly reject credentials,
    unsupported schemes, blank hosts, and control characters.
    """
    raise NotImplementedError("TODO: validate an HTTP(S) URL")


def check_link(source: WebSourceSpec) -> LinkCheckResult:
    """Check one configured source without raising for an HTTP failure.

    TODO: validate first; issue a bounded request with redirects and the configured
    user agent; fall back from HEAD to a streamed GET when HEAD is unsupported;
    close the response promptly; and capture errors in the returned record.
    """
    raise NotImplementedError("TODO: check one supplementary source link")


def check_all_links(sources: list[WebSourceSpec]) -> list[LinkCheckResult]:
    """Return one link-check result per configured source.

    TODO: reject duplicate IDs, check politely with bounded concurrency or
    pacing, preserve input order, and retain failures for the final report.
    """
    raise NotImplementedError("TODO: check all supplementary links")


def main() -> None:
    """CLI declaration for coverage-link QA.

    TODO: check ``config.SUPPLEMENTARY_SOURCES``, print a compact summary, return
    a nonzero process status when any URL is invalid, and optionally save CSV.
    Do not execute network requests when this module is merely imported.
    """
    raise NotImplementedError("TODO: implement the link-check CLI")


if __name__ == "__main__":
    main()
