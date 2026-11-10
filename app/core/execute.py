"""Write the plan back to Bilibili.

One action only: move a video from the folder it sits in now into the folder
that matches its planned category. Nothing is deleted and nothing is renamed.
A run can leave source folders empty; those are offered for deletion afterwards,
on a confirmation, and only when a fresh read of the account shows them empty.

Order matters. Folders are created (or reused) first, because a move needs a
real target id. Items are then grouped by (source, target) so one request can
carry up to BATCH videos; the endpoint takes a list.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Callable

from ..adapters.bili import BATCH_MOVE, BiliClient, BiliError, NotLoggedIn, RateLimited
from . import store, taxonomy

Progress = Callable[[str, int, int, str], None]

# A throttled batch is retried after these waits, in seconds. Only a refusal is
# retried: the server rejected the request before applying it, so repeating it
# cannot move the same videos twice. A timeout is not retried, because by then
# the move may well have happened.
RATE_WAITS = (8, 25)

# Appended when the name a category wants is already taken by a folder this run
# did not make. Two selections that were never meant to be one should not end
# up in the same folder just because they share a word.
AI_SUFFIX = "-AI分类"


class ExecuteError(RuntimeError):
    pass


class Interrupted(RuntimeError):
    """The run stopped part-way, and knows exactly how far it got.

    A second run picks up from here: a row is marked 移动 only after its calls
    returned, so nothing that already went through is attempted again.
    """

    def __init__(
        self,
        reason: str,
        *,
        moved: int,
        remaining: int,
        batches: int,
        created: list[str],
        reused: list[str],
        unusable: int,
        login_lost: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.moved = moved
        self.remaining = remaining
        self.batches = batches
        self.created = created
        self.reused = reused
        self.unusable = unusable
        self.login_lost = login_lost

    def as_summary(self) -> dict:
        return {
            "moved": self.moved,
            "remaining": self.remaining,
            "batches": self.batches,
            "created": self.created,
            "reused": self.reused,
            "unusable": self.unusable,
            "stopped": self.reason,
        }


def plan_summary(conn: sqlite3.Connection) -> dict:
    """What a run would do, without doing it."""
    rows = conn.execute(
        "SELECT target, COUNT(*) AS n FROM plan_item "
        "WHERE plan_id = 'current' AND action = 'move' GROUP BY target"
    ).fetchall()
    if not rows:
        done = conn.execute(
            "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current'"
        ).fetchone()[0]
        if done:
            raise ExecuteError("这份归类已经全部执行过了；重新归类后再执行")
        raise ExecuteError("还没有归类结果，先点「开始归类」")

    total = sum(row["n"] for row in rows)
    moved = conn.execute(
        "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current' AND action = 'moved'"
    ).fetchone()[0]
    return {
        "total": total,
        "moved": moved,
        "targets": [{"name": row["target"], "count": row["n"]} for row in rows],
    }


def run(conn: sqlite3.Connection, client: BiliClient, progress: Progress | None = None) -> dict:
    summary = plan_summary(conn)
    total = summary["total"]
    # Folder names come from what is still to do, not from the plan: on a
    # resume, the targets that finished need no folder work.
    targets = [t["name"] for t in summary["targets"]]

    moved = 0
    done_batches = 0
    total_batches = 0
    created: list[str] = []
    reused: list[str] = []
    unusable = 0

    def note(text: str) -> None:
        if progress:
            progress("execute", done_batches, total_batches, text)

    try:
        intros = {
            taxonomy.display_name(c): (c.get("summary") or c.get("definition") or "")[:80]
            for c in (taxonomy.load(conn) or [])
        }
        existing = _folder_index(client)
        ours = store.created_names(conn)
        folder_ids: dict[str, int] = {}
        for name in targets:
            actual = _folder_name(name, existing, ours)
            found = existing.get(actual)
            if found is not None:
                folder_ids[name] = found
                reused.append(actual)
                continue
            # The folder carries the description written from its contents, so
            # B站 itself shows what the folder is for.
            folder_id = client.create_folder(actual, intro=intros.get(name, ""))
            folder_ids[name] = folder_id
            # Recorded so the next run in this session reuses it instead of
            # opening a second 「-AI分类」 beside it.
            store.upsert_container(
                conn, "bilibili", "favorites", str(folder_id), actual, 0, False,
                created=True,
            )
            created.append(actual)

        groups, unusable = _group_moves(conn, folder_ids)
        total_batches = sum(math.ceil(len(pairs) / BATCH_MOVE) for pairs in groups.values())

        # A video can sit in two folders. It counts as done only once every
        # membership has gone through, otherwise a resume would skip the
        # second one and leave the video behind in the old folder.
        left: dict[str, int] = {}
        for pairs in groups.values():
            for _, item_id in pairs:
                left[item_id] = left.get(item_id, 0) + 1

        for (src, tar), pairs in groups.items():
            for start_at in range(0, len(pairs), BATCH_MOVE):
                chunk = pairs[start_at : start_at + BATCH_MOVE]
                _move(client, src, tar, [aid for aid, _ in chunk], note)
                for _, item_id in chunk:
                    left[item_id] -= 1
                finished = [
                    item_id
                    for item_id in dict.fromkeys(item_id for _, item_id in chunk)
                    if left[item_id] == 0
                ]
                # Mark only after the call returns: a batch that failed must
                # stay movable on the next run.
                _mark_moved(conn, finished)
                moved += len(finished)
                done_batches += 1
                if progress:
                    progress(
                        "execute",
                        done_batches,
                        total_batches,
                        f"已移动 {moved}/{total} 条 · 第 {done_batches}/{total_batches} 批",
                    )
    except BiliError as exc:
        raise Interrupted(
            _pause_reason(exc),
            moved=moved,
            remaining=_pending(conn),
            batches=done_batches,
            created=created,
            reused=reused,
            unusable=unusable,
            login_lost=isinstance(exc, NotLoggedIn),
        ) from exc

    # Folders this run pulled everything out of. Offered, not deleted: the
    # page lists them and the person confirms. A resumed run reports the ones
    # earlier attempts emptied, which is exactly what should be cleaned up.
    empty = _empty_sources(
        client,
        sources={src for src, _ in groups},
        keep=set(folder_ids.values()),
    )

    return {
        "moved": moved,
        "remaining": _pending(conn),
        "batches": done_batches,
        "created": created,
        "reused": reused,
        "unusable": unusable,
        "empty": empty,
        "targets": len(targets),
    }


def _move(
    client: BiliClient,
    src: int,
    tar: int,
    aids: list[int],
    note: Callable[[str], None],
) -> None:
    """Move one batch, waiting out a throttle instead of giving up on it."""
    waits = list(RATE_WAITS)
    for attempt in range(len(waits) + 1):
        if attempt:
            wait = waits[attempt - 1]
            note(f"被限流，{wait} 秒后重试（第 {attempt}/{len(waits)} 次）")
            time.sleep(wait)
        try:
            client.move_items(src, tar, aids)
            return
        except RateLimited:
            if attempt == len(waits):
                raise


def _pending(conn: sqlite3.Connection) -> int:
    """Videos still waiting to move. One row per video, whatever its folders."""
    return conn.execute(
        "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current' AND action = 'move'"
    ).fetchone()[0]


def _pause_reason(exc: BiliError) -> str:
    if isinstance(exc, NotLoggedIn):
        return "登录已失效，已停止"
    if isinstance(exc, RateLimited):
        return "B 站限流，已停止；隔 10-30 分钟再点「继续执行」"
    return f"接口报错，已停止：{exc}"

def _empty_sources(
    client: BiliClient, sources: set[int], keep: set[int]
) -> list[dict]:
    """Source folders that are provably empty now.

    Re-read straight from the account: the local tables still describe the
    layout from before the moves. Anything that cannot be proven empty is left
    out, because the next step deletes it.
    """
    try:
        data = client.folders() or {}
    except BiliError:
        return []

    out: list[dict] = []
    for folder in data.get("list") or []:
        try:
            folder_id = int(folder.get("id"))
        except (TypeError, ValueError):
            continue
        # The default folder cannot be deleted, and a destination is a folder
        # that is supposed to exist even when nothing landed in it.
        if folder_id == 0 or folder_id in keep or folder_id not in sources:
            continue
        count = folder.get("media_count")
        if isinstance(count, int) and count == 0:
            out.append({"id": folder_id, "name": (folder.get("title") or "").strip()})
    return out


def _folder_name(name: str, existing: dict[str, int], ours: set[str]) -> str:
    """The name this run will actually use for a category.

    A taken name belongs to the person unless this tool made that folder: its
    contents are a selection of their own and would be mixed with a different
    one. Such a name gets AI_SUFFIX instead.

    The suffixed name is checked first, so a folder left by an earlier run is
    reused rather than duplicated on every run.
    """
    alt = name + AI_SUFFIX
    if alt in existing:
        return alt
    if name in existing and name not in ours:
        return alt
    return name


def _folder_index(client: BiliClient) -> dict[str, int]:
    data = client.folders() or {}
    index: dict[str, int] = {}
    for folder in data.get("list") or []:
        title = (folder.get("title") or "").strip()
        if title:
            index[title] = int(folder.get("id"))
    return index


def _mark_moved(conn: sqlite3.Connection, item_ids: list[str]) -> None:
    """Record success per batch, so a retry resumes instead of repeating."""
    conn.executemany(
        "UPDATE plan_item SET action = 'moved' WHERE plan_id = 'current' AND item_id = ?",
        [(item_id,) for item_id in item_ids],
    )
    conn.commit()


def _group_moves(
    conn: sqlite3.Connection, folder_ids: dict[str, int]
) -> tuple[dict[tuple[int, int], list[tuple[int, str]]], int]:
    """Group video ids by (source folder, target folder).

    A video can sit in more than one folder; each membership is its own move.
    Items already in their target are skipped, and items whose stored payload
    has no aid cannot be moved by this endpoint.
    """
    rows = conn.execute(
        """
        SELECT p.item_id, p.target, m.container_id AS src, i.meta_json
          FROM plan_item p
          JOIN item i
            ON i.platform = 'bilibili' AND i.kind = 'video' AND i.item_id = p.item_id
          JOIN membership m
            ON m.item_kind = 'video' AND m.container_kind = 'favorites'
           AND m.item_id = p.item_id
          LEFT JOIN container c
            ON c.platform = m.platform AND c.kind = m.container_kind
           AND c.container_id = m.container_id
         WHERE p.plan_id = 'current' AND p.action = 'move'
           -- an excluded folder is never a source, even if a plan predates it
           AND COALESCE(c.keep_out, 0) = 0
        """
    ).fetchall()

    groups: dict[tuple[int, int], list[tuple[int, str]]] = {}
    unusable = 0
    for row in rows:
        target = folder_ids.get(row["target"])
        if target is None:
            unusable += 1
            continue
        try:
            src = int(row["src"])
        except (TypeError, ValueError):
            unusable += 1
            continue
        if src == target:
            continue

        aid = _aid(row["meta_json"])
        if aid is None:
            unusable += 1
            continue
        groups.setdefault((src, target), []).append((aid, row["item_id"]))

    return groups, unusable


def _aid(meta_json: str | None) -> int | None:
    """The favourite list returns `id` (the aid) alongside `bvid`."""
    if not meta_json:
        return None
    try:
        meta = json.loads(meta_json)
    except ValueError:
        return None
    value = meta.get("id")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
