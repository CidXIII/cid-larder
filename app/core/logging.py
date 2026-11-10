"""File logging for the command-line entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import DATA_DIR


def setup(name: str) -> Path:
    """Log DEBUG to data/<name>.log; only warnings reach the console."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    root.addHandler(console)

    # Third-party loggers are noisy and may include request details.
    for noisy in ("httpcore", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return path
