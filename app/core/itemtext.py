"""One line of text per item, built from everything the favourites API gives.

Used by both the category suggestion (what the model reads) and the embedding
step (what gets vectorised), so the two never drift apart.
"""

from __future__ import annotations

import time

INTRO_LIMIT = 32


def upper_name(meta: dict) -> str:
    """The uploader's display name, or empty. `upper` is sometimes absent."""
    upper = meta.get("upper")
    if isinstance(upper, dict):
        return _clean(upper.get("name"))
    return ""


def describe(title: str, meta: dict) -> str:
    parts = [_clean(title)]

    upper = upper_name(meta)
    if upper:
        parts.append(f"UP主：{upper}")

    seconds = _int(meta.get("duration"))
    if seconds:
        parts.append(f"{seconds // 60} 分钟")

    pages = _int(meta.get("page"))
    if pages > 1:
        parts.append(f"{pages}P")

    stamp = _int(meta.get("ctime")) or _int(meta.get("pubtime"))
    if stamp:
        parts.append(time.strftime("%Y-%m", time.localtime(stamp)))

    intro = _clean(meta.get("intro"))[:INTRO_LIMIT]
    if intro:
        parts.append(f"简介：{intro}")

    collection = _collection(meta.get("ugc"))
    if collection:
        parts.append(f"合集：{collection}")

    return " ｜ ".join(part for part in parts if part)


def _collection(ugc: object) -> str:
    """`ugc` is undocumented; read it defensively."""
    if isinstance(ugc, dict):
        for key in ("title", "name", "season_title"):
            value = ugc.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(ugc, str) and ugc.strip():
        return ugc.strip()
    return ""


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
