#!/usr/bin/env python
"""Collect Bilibili favourites into the local database.

    python scripts/collect.py

Read-only against the platform. Re-running refreshes titles and timestamps
and rewrites membership; it never removes an item that left a folder.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adapters.bili import BiliClient, BiliError, NotLoggedIn, RateLimited  # noqa: E402
from app.core import logging as logsetup  # noqa: E402
from app.core import store  # noqa: E402
from app.core.collector import collect  # noqa: E402
from app.core.config import DATA_DIR  # noqa: E402
from app.core.db import connect, init_db  # noqa: E402
from app.core.session import acquire_session  # noqa: E402

# Requests start at least this far apart. Small enough to be quick, wide
# enough that three in flight do not look like a burst.
REQUEST_INTERVAL = 0.35
CONCURRENCY = 3


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    logsetup.setup("collect")
    logging.getLogger("collect").info("start")

    print("Cid-Larder 采集 —— 只读拉取，不会修改平台上的任何内容。")
    login = acquire_session()
    if login is None:
        print(f"\n没拿到登录态。日志：{DATA_DIR / 'collect.log'}")
        return 2

    conn = connect()
    try:
        init_db(conn)
        print("\n数据库：内存（结束即清空）")
        print("开始采集（每页 1-2 秒，可以随时中断，下次重跑会补齐）：\n")

        reporter = Reporter()
        with BiliClient(
            sessdata=login.sessdata,
            csrf=login.csrf,
            uid=login.uid,
            min_interval=REQUEST_INTERVAL,
            jitter=0.2,
        ) as client:
            summary = collect(client, conn, progress=reporter, concurrency=CONCURRENCY)
        reporter.close()

        print("\n" + "-" * 52)
        print(summary.report())
        totals = store.counts(conn)
        print(f"库内：容器 {totals['container']} · 条目 {totals['item']} · 归属 {totals['membership']}")
        print("\n采集完成。下一步是分类，那一层还没写。")
        return 0
    finally:
        conn.close()


class Reporter:
    """One updating line per stage, so a long run never looks frozen."""

    WIDTH = 88

    def __init__(self) -> None:
        self.stage: str | None = None

    def __call__(self, stage: str, done: int, total: int, note: str) -> None:
        if stage != self.stage:
            if self.stage is not None:
                print()
            self.stage = stage
        text = f"[{done}/{total}] {note}" if total else note
        print("\r" + _fit(text, self.WIDTH), end="", flush=True)

    def close(self) -> None:
        if self.stage is not None:
            print()


def _fit(text: str, width: int) -> str:
    """Pad or trim to a fixed display width, counting CJK as two columns."""
    out, used = [], 0
    for ch in text:
        step = 2 if ord(ch) > 0x2E7F else 1
        if used + step > width:
            break
        out.append(ch)
        used += step
    return "".join(out) + " " * (width - used)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotLoggedIn as exc:
        print(f"\n登录态失效：{exc}")
        raise SystemExit(1) from exc
    except RateLimited as exc:
        print(f"\n触发风控：{exc}\n停 10 分钟再跑一次。已采集的部分保留在库里。")
        raise SystemExit(1) from exc
    except BiliError as exc:
        print(f"\n接口报错：{exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\n已中断。已采集的部分保留在库里，重跑会继续补齐。")
        raise SystemExit(130) from None
