"""Token accounting.

Every call that can report usage writes one row. The providers disagree on field
names and some omit usage entirely, so a missing number is stored as NULL and
surfaced as "not reported" rather than being counted as zero.
"""

from __future__ import annotations

import sqlite3
import time

STAGES = {
    "suggest": "生成分类建议",
    "classify": "向量归类",
}


def record(
    conn: sqlite3.Connection,
    stage: str,
    provider: str,
    model: str,
    usage: dict | None,
) -> None:
    """Store one call. `usage` is the raw dict from the API response."""
    usage = usage if isinstance(usage, dict) else {}
    prompt = _pick(usage, "prompt_tokens", "input_tokens")
    completion = _pick(usage, "completion_tokens", "output_tokens")
    total = _pick(usage, "total_tokens")
    cached = _cached(usage)

    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)

    conn.execute(
        """
        INSERT INTO usage_log
            (stage, provider, model, prompt_tokens, completion_tokens,
             total_tokens, cached_tokens, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (stage, provider, model, prompt, completion, total, cached, int(time.time())),
    )
    conn.commit()


def summary(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """
        SELECT stage,
               COUNT(*)                                        AS calls,
               SUM(prompt_tokens)                              AS prompt_tokens,
               SUM(completion_tokens)                          AS completion_tokens,
               SUM(total_tokens)                               AS total_tokens,
               SUM(cached_tokens)                             AS cached_tokens,
               SUM(CASE WHEN total_tokens IS NULL THEN 1 ELSE 0 END) AS missing
          FROM usage_log
         GROUP BY stage
         ORDER BY calls DESC
        """
    ).fetchall()

    stages = [
        {
            "stage": row["stage"],
            "label": STAGES.get(row["stage"], row["stage"]),
            "calls": row["calls"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "cached_tokens": row["cached_tokens"],
            "missing": row["missing"],
        }
        for row in rows
    ]

    return {"stages": stages, "total": _totals(stages)}


def watermark(conn: sqlite3.Connection) -> int:
    """Highest row id right now. Pass it to `since` to isolate one operation."""
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM usage_log").fetchone()
    return int(row[0] or 0)


def since(conn: sqlite3.Connection, after_id: int) -> dict:
    """Tokens burnt by everything logged after `after_id`.

    One user action can span several calls, and the same stage may have run
    before. Counting rows newer than the watermark is what makes "this run"
    mean this run instead of the running total.
    """
    row = conn.execute(
        """
        SELECT COUNT(*)                   AS calls,
               SUM(prompt_tokens)         AS prompt_tokens,
               SUM(completion_tokens)     AS completion_tokens,
               SUM(total_tokens)          AS total_tokens,
               SUM(cached_tokens)         AS cached_tokens,
               SUM(CASE WHEN total_tokens IS NULL THEN 1 ELSE 0 END) AS missing
          FROM usage_log
         WHERE id > ?
        """,
        (after_id,),
    ).fetchone()
    return _row(row)


def _row(row) -> dict:
    if row is None:
        return {"calls": 0, "prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "cached_tokens": None, "missing": 0}
    return {
        "calls": row["calls"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "cached_tokens": row["cached_tokens"],
        "missing": row["missing"],
    }


def _totals(stages: list[dict]) -> dict:
    if not stages:
        return {"calls": 0, "prompt_tokens": None, "completion_tokens": None,
                "total_tokens": None, "cached_tokens": None, "missing": 0}
    return {
        "calls": sum(s["calls"] for s in stages),
        "prompt_tokens": _sum_or_none(s["prompt_tokens"] for s in stages),
        "completion_tokens": _sum_or_none(s["completion_tokens"] for s in stages),
        "total_tokens": _sum_or_none(s["total_tokens"] for s in stages),
        "cached_tokens": _sum_or_none(s["cached_tokens"] for s in stages),
        "missing": sum(s["missing"] for s in stages),
    }


def _sum_or_none(values) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _cached(usage: dict) -> int | None:
    """DeepSeek reports both a top-level field and a details object."""
    direct = usage.get("prompt_cache_hit_tokens")
    if isinstance(direct, (int, float)):
        return int(direct)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        value = details.get("cached_tokens")
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _pick(usage: dict, *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return None
