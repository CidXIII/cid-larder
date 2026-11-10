"""Read-only views over the collected library."""

from __future__ import annotations

import json
import sqlite3

from .itemtext import upper_name

PLATFORM = "bilibili"


def overview(conn: sqlite3.Connection) -> dict:
    """What is in the database right now, grouped the way the platforms are."""
    favorites = _containers(conn, "favorites")

    videos = conn.execute(
        "SELECT COUNT(*) FROM item WHERE platform = ? AND kind = 'video'", (PLATFORM,)
    ).fetchone()[0]

    return {
        "platform": PLATFORM,
        "favorites": favorites,
        "totals": {"videos": videos, "containers": len(favorites)},
    }


def _containers(conn: sqlite3.Connection, kind: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.container_id,
               c.name,
               c.item_count,
               c.is_default,
               c.keep_out,
               (SELECT COUNT(*) FROM membership m
                 WHERE m.platform = c.platform
                   AND m.container_kind = c.kind
                   AND m.container_id = c.container_id) AS stored
          FROM container c
         WHERE c.platform = ? AND c.kind = ?
         ORDER BY stored DESC, c.name
        """,
        (PLATFORM, kind),
    ).fetchall()

    return [
        {
            "id": row["container_id"],
            "name": row["name"] or "（未命名）",
            "reported": row["item_count"],
            "stored": row["stored"],
            "default": bool(row["is_default"]),
            "excluded": bool(row["keep_out"]),
        }
        for row in rows
    ]


def items_in(conn: sqlite3.Connection, container_id: str, limit: int = 200) -> dict:
    """Titles and uploaders inside one folder, capped for the browser."""
    total = conn.execute(
        """
        SELECT COUNT(*) FROM membership
         WHERE platform = ? AND container_kind = 'favorites' AND container_id = ?
        """,
        (PLATFORM, container_id),
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT i.item_id, i.title, i.meta_json
          FROM membership m
          JOIN item i ON i.platform = m.platform
                     AND i.kind = m.item_kind
                     AND i.item_id = m.item_id
         WHERE m.platform = ? AND m.container_kind = 'favorites'
           AND m.container_id = ?
         ORDER BY COALESCE(m.position, 1000000000), i.title
         LIMIT ?
        """,
        (PLATFORM, container_id, limit),
    ).fetchall()

    return {
        "total": total,
        "items": [_brief(row["item_id"], row["title"], row["meta_json"]) for row in rows],
    }


def _brief(item_id: str, title: str | None, meta_json: str | None) -> dict:
    try:
        meta = json.loads(meta_json or "{}")
    except ValueError:
        meta = {}
    return {
        "id": item_id,
        "title": (title or "").strip() or "（无标题）",
        "upper": upper_name(meta),
    }
