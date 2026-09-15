"""Discover resume files in the input directory.

Kept deliberately small: find candidate files, ignore anything we can't parse,
and never raise on a bad path so the batch always gets a chance to run.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def discover_resumes(input_dir: str | Path) -> list[Path]:
    """Return supported resume files under ``input_dir``, sorted by name.

    Recurses into subdirectories. Non-resume files, hidden files, and empty
    files are skipped. A missing or non-directory path yields an empty list
    rather than an exception.
    """

    root = Path(input_dir)
    if not root.exists():
        logger.warning("Input path does not exist: %s", root)
        return []
    if not root.is_dir():
        logger.warning("Input path is not a directory: %s", root)
        return []

    resumes: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if path.stat().st_size == 0:
                logger.warning("Skipping empty file: %s", path.name)
                continue
        except OSError as exc:
            logger.warning("Could not stat %s: %s", path.name, exc)
            continue
        resumes.append(path)

    logger.info("Discovered %d resume file(s) in %s", len(resumes), root)
    return resumes
