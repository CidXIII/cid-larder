"""Writes to SQLite. Plain functions; the caller owns the transaction."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable


def now() -> int:
    return int(time.time())


def upsert_container(
    conn: sqlite3.Connection,
    platform: str,
    kind: str,
    container_id: str,
    name: str | None,
    item_count: int | None = None,
    is_default: bool = False,
    created: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO container
            (platform, kind, container_id, name, item_count, is_default,
             created, seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(platform, kind, container_id) DO UPDATE SET
            name       = excluded.name,
            item_count = excluded.item_count,
            is_default = excluded.is_default,
            -- never un-mark a folder this tool made: a later collect must not
            -- turn it back into a stranger's folder
            created    = MAX(container.created, excluded.created),
            seen_at    = excluded.seen_at
        """,
        (
            platform, kind, container_id, name, item_count, int(is_default),
            int(created), now(),
        ),
    )


def set_keep_out(
    conn: sqlite3.Connection,
    container_id: str,
    keep_out: bool,
    platform: str = "bilibili",
    kind: str = "favorites",
) -> int:
    """Mark one folder as not taking part in the next run.

    Nothing on Bilibili changes: this only stops the folder's items from being
    read into a new classification, and stops execution from using it as a
    source.
    """
    cur = conn.execute(
        """
        UPDATE container SET keep_out = ?
         WHERE platform = ? AND kind = ? AND container_id = ?
        """,
        (int(keep_out), platform, kind, str(container_id)),
    )
    conn.commit()
    return cur.rowcount


def created_names(
    conn: sqlite3.Connection, platform: str = "bilibili", kind: str = "favorites"
) -> set[str]:
    """Folders this process made, so a second run reuses instead of re-making."""
    rows = conn.execute(
        "SELECT name FROM container WHERE platform = ? AND kind = ? AND created = 1",
        (platform, kind),
    ).fetchall()
    return {row["name"] for row in rows if row["name"]}


def upsert_item(
    conn: sqlite3.Connection,
    platform: str,
    kind: str,
    item_id: str,
    title: str | None,
    meta: dict | None,
    seen_at: int,
) -> bool:
    """Insert or refresh one item. Returns True when it was not seen before."""
    payload = json.dumps(meta, ensure_ascii=False) if meta is not None else None

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO item
            (platform, kind, item_id, title, meta_json, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (platform, kind, item_id, title, payload, seen_at, seen_at),
    )
    created = cursor.rowcount == 1

    conn.execute(
        """
        UPDATE item
           SET title     = ?,
               meta_json = ?,
               last_seen = ?
         WHERE platform = ? AND kind = ? AND item_id = ?
        """,
        (title, payload, seen_at, platform, kind, item_id),
    )
    return created


def replace_membership(
    conn: sqlite3.Connection,
    platform: str,
    container_kind: str,
    container_id: str,
    item_kind: str,
    members: Iterable[tuple[str, int | None, int | None]],
) -> int:
    """Rewrite one container's membership. Returns how many rows were written.

    Rewriting rather than merging is what makes a re-run converge: items that
    left the folder disappear from `membership` but stay in `item`.
    """
    conn.execute(
        """
        DELETE FROM membership
         WHERE platform = ? AND container_kind = ? AND container_id = ?
        """,
        (platform, container_kind, container_id),
    )
    rows = [
        (platform, container_kind, container_id, item_kind, i, p, a)
        for i, p, a in members
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO membership
            (platform, container_kind, container_id, item_kind, item_id, position, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def forget_container(
    conn: sqlite3.Connection, platform: str, kind: str, container_id: str
) -> None:
    """Drop a container the account no longer has.

    Only used after the folder was really deleted upstream; the membership
    rows go first, though a folder worth deleting has none left.
    """
    conn.execute(
        "DELETE FROM membership WHERE platform = ? AND container_kind = ? AND container_id = ?",
        (platform, kind, container_id),
    )
    conn.execute(
        "DELETE FROM container WHERE platform = ? AND kind = ? AND container_id = ?",
        (platform, kind, container_id),
    )
    conn.commit()


def counts(conn: sqlite3.Connection, platform: str | None = None) -> dict[str, int]:
    where, params = ("WHERE platform = ?", (platform,)) if platform else ("", ())
    out: dict[str, int] = {}
    for table in ("container", "item", "membership"):
        out[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} {where}", params
        ).fetchone()[0]
    return out
