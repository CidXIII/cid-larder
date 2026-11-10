"""Collect Bilibili favourites into SQLite.

Read-only against the platform. Safe to run repeatedly: re-running refreshes
titles and timestamps and rewrites membership, and never deletes an item just
because it left a folder.

Each container commits on its own, so an interrupted run keeps its progress.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..adapters.bili import (
    FAV_PAGE_SIZE,
    PROBE_PAGE_SIZE,
    BiliClient,
    BiliError,
    RateLimited,
)
from . import store

log = logging.getLogger(__name__)

PLATFORM = "bilibili"
MAX_PAGES = 1000          # runaway guard, never a real limit
BACKOFF_SECONDS = 10.0    # pause after risk control before retrying once

Progress = Callable[[str, int, int, str], None]


@dataclass
class Summary:
    containers: int = 0
    videos: int = 0
    videos_new: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "containers": self.containers,
            "videos": self.videos,
            "seconds": round(self.seconds, 1),
        }

    def report(self) -> str:
        return (
            f"收藏 {self.videos} 条（新增 {self.videos_new}）\n"
            f"用时 {self.seconds:.1f}s"
        )


def collect(
    client: BiliClient,
    conn: sqlite3.Connection,
    progress: Progress | None = None,
    concurrency: int = 3,
) -> Summary:
    started = time.monotonic()
    summary = Summary()
    seen_at = store.now()

    _collect_favorites(client, conn, summary, seen_at, progress, concurrency)

    summary.seconds = time.monotonic() - started
    return summary


def _collect_favorites(
    client: BiliClient,
    conn: sqlite3.Connection,
    summary: Summary,
    seen_at: int,
    progress: Progress | None,
    concurrency: int,
) -> None:
    folders = (client.folders() or {}).get("list") or []

    for index, folder in enumerate(folders, 1):
        media_id = folder.get("id")
        if media_id is None:
            continue

        label = str(folder.get("title") or "")
        container_id = str(media_id)

        try:
            store.upsert_container(
                conn, PLATFORM, "favorites", container_id,
                name=folder.get("title"),
                item_count=folder.get("media_count"),
                is_default=(media_id == 0),
            )
            summary.containers += 1

            def on_page(done: int, count: int) -> None:
                if progress:
                    progress(
                        "favorites", index, len(folders),
                        f"{label} · {done} 页 · {count} 条",
                    )

            items, _pages = _fetch_items(client, media_id, concurrency, on_page)

            members: list[tuple[str, int | None, int | None]] = []
            for position, entry in enumerate(items):
                item_id = str(entry.get("bvid") or entry.get("id") or "")
                if not item_id:
                    continue

                created = store.upsert_item(
                    conn, PLATFORM, "video", item_id, entry.get("title"), entry, seen_at
                )
                summary.videos += 1
                if created:
                    summary.videos_new += 1
                members.append((item_id, position, entry.get("fav_time")))

            store.replace_membership(
                conn, PLATFORM, "favorites", container_id, "video", members
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        if progress:
            progress("favorites", index, len(folders), f"{label} · {len(items)} 条")


def _fetch_items(
    client: BiliClient,
    media_id: int,
    concurrency: int,
    on_page: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], int]:
    """Fetch a whole folder. Returns (items, pages_requested).

    The first request asks for more than expected: if the server still has
    more to give, the count it returned is its real page-size cap.
    """
    try:
        first = client.folder_items(media_id, pn=1, ps=PROBE_PAGE_SIZE) or {}
    except RateLimited:
        raise
    except BiliError:
        # An oversized `ps` may be rejected outright rather than capped.
        log.info("page-size probe rejected for media %s; using %s", media_id, FAV_PAGE_SIZE)
        first = client.folder_items(media_id, pn=1, ps=FAV_PAGE_SIZE) or {}

    items = list(first.get("medias") or [])
    pages = 1
    if on_page:
        on_page(pages, len(items))

    if not first.get("has_more"):
        return items, pages

    page_size = len(items) or FAV_PAGE_SIZE
    page = 2
    while page <= MAX_PAGES:
        batch = list(range(page, page + max(1, concurrency)))
        try:
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                results = list(pool.map(
                    lambda p: client.folder_items(media_id, pn=p, ps=page_size) or {},
                    batch,
                ))
        except RateLimited:
            # Back off once, then continue one request at a time.
            log.warning("rate limited at page %s of media %s; backing off", page, media_id)
            interval = getattr(client, "min_interval", 1.0)
            client.min_interval = min(interval * 2 + 0.5, 5.0)
            time.sleep(BACKOFF_SECONDS)
            results = [client.folder_items(media_id, pn=p, ps=page_size) or {} for p in batch]

        got_any = False
        for data in results:
            chunk = data.get("medias") or []
            if chunk:
                got_any = True
                items.extend(chunk)

        pages += len(batch)
        if on_page:
            on_page(pages, len(items))

        if not got_any or not results[-1].get("has_more"):
            break
        page += len(batch)

    return items, pages
