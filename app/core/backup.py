"""A snapshot the user can keep.

Nothing here is written to disk by the server. The JSON goes straight to the
browser as a download, so the "no data on disk" rule still holds: the file
lands wherever the user chooses.

What it captures is the shape of the library -- which container held which
item, in what order -- because that is what you need to undo a move. Titles
are included for legibility; raw API payloads are not.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

FORMAT_VERSION = 1


def dump(conn: sqlite3.Connection) -> dict:
    containers = [
        {
            "platform": row["platform"],
            "kind": row["kind"],
            "id": row["container_id"],
            "name": row["name"],
            "item_count": row["item_count"],
            "is_default": bool(row["is_default"]),
        }
        for row in conn.execute(
            "SELECT * FROM container ORDER BY kind, container_id"
        )
    ]

    items = [
        {
            "platform": row["platform"],
            "kind": row["kind"],
            "id": row["item_id"],
            "title": row["title"],
        }
        for row in conn.execute("SELECT platform, kind, item_id, title FROM item")
    ]

    membership = [
        {
            "container_kind": row["container_kind"],
            "container_id": row["container_id"],
            "item_kind": row["item_kind"],
            "item_id": row["item_id"],
            "position": row["position"],
        }
        for row in conn.execute(
            """
            SELECT container_kind, container_id, item_kind, item_id, position
              FROM membership
             ORDER BY container_kind, container_id, position
            """
        )
    ]

    plan = [
        {
            "item_id": row["item_id"],
            "target": row["target"],
            "confidence": row["confidence"],
            "action": row["action"],
        }
        for row in conn.execute(
            "SELECT item_id, target, confidence, action FROM plan_item WHERE plan_id = 'current'"
        )
    ]

    # Whether this snapshot was taken before anything moved or after a
    # partial run. Without it two files look identical, and the only one worth
    # restoring from is the one with moved = 0.
    progress = {
        "moved": conn.execute(
            "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current' AND action = 'moved'"
        ).fetchone()[0],
        "remaining": conn.execute(
            "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current' AND action = 'move'"
        ).fetchone()[0],
    }

    return {
        "format": FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "progress": progress,
        "counts": {
            "containers": len(containers),
            "items": len(items),
            "membership": len(membership),
            "planned": len(plan),
        },
        "containers": containers,
        "items": items,
        "membership": membership,
        "plan": plan,
    }
