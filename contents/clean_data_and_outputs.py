"""Remove generated data and model artifacts while preserving folder structure."""

from __future__ import annotations

import shutil
from pathlib import Path

from config import DATA_PROCESSED, DATA_RAW, OUTPUTS, PROJECT_DIR


PRESERVED_NAMES = {".gitkeep"}
CLEANUP_DIRECTORIES = (DATA_RAW, DATA_PROCESSED, OUTPUTS)


def _validate_cleanup_directory(directory: Path) -> Path:
    """Return a safe configured directory or raise before deleting anything."""
    resolved_project = PROJECT_DIR.resolve()
    resolved_directory = directory.resolve()
    allowed_directories = {
        DATA_RAW.resolve(),
        DATA_PROCESSED.resolve(),
        OUTPUTS.resolve(),
    }

    if directory.is_symlink():
        raise ValueError(f"Refusing to clean a symbolic link: {directory}")
    if resolved_directory not in allowed_directories:
        raise ValueError(f"Directory is not an approved cleanup target: {directory}")
    if resolved_project not in resolved_directory.parents:
        raise ValueError(f"Cleanup directory is outside the project: {directory}")
    if resolved_directory == resolved_project:
        raise ValueError("Refusing to clean the project directory itself")

    return resolved_directory


def clean_directory(directory: Path) -> tuple[int, int]:
    """Delete one directory's contents except preserved placeholder files.

    Return ``(files_removed, directories_removed)``.
    """
    safe_directory = _validate_cleanup_directory(directory)
    safe_directory.mkdir(parents=True, exist_ok=True)

    files_removed = 0
    directories_removed = 0
    for entry in safe_directory.iterdir():
        if entry.name in PRESERVED_NAMES:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
            files_removed += 1
        elif entry.is_dir():
            shutil.rmtree(entry)
            directories_removed += 1
        else:
            raise ValueError(f"Unsupported filesystem entry: {entry}")

    return files_removed, directories_removed


def main() -> None:
    """Clean all configured generated-data directories without arguments."""
    total_files = 0
    total_directories = 0

    for directory in CLEANUP_DIRECTORIES:
        files_removed, directories_removed = clean_directory(directory)
        total_files += files_removed
        total_directories += directories_removed
        print(
            f"Cleaned {directory}: removed {files_removed} files and "
            f"{directories_removed} directories"
        )

    print(
        f"Cleanup complete: removed {total_files} files and "
        f"{total_directories} directories; preserved .gitkeep files"
    )


if __name__ == "__main__":
    main()
