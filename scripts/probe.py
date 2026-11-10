#!/usr/bin/env python
"""P0 probe: read-only checks against the live Bilibili API.

    python scripts/probe.py

Prints one line per check as it finishes; a few throttled requests otherwise
look like a hang. Nothing is modified.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adapters.bili import BiliClient, BiliError, NotLoggedIn, RateLimited  # noqa: E402
from app.core import logging as logsetup  # noqa: E402
from app.core.config import DATA_DIR  # noqa: E402
from app.core.session import acquire_session  # noqa: E402

EXPECTED_ITEM_FIELDS = [
    "id", "type", "title", "cover", "intro", "page", "duration",
    "upper", "attr", "cnt_info", "bvid", "pubtime", "fav_time",
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    logsetup.setup("probe")
    logging.getLogger("probe").info("start")

    print("Cid-Larder 探针 —— 只读检查，不会修改任何收藏。")
    login = acquire_session()
    if login is None:
        print(f"\n没拿到登录态。日志：{DATA_DIR / 'probe.log'}")
        return 2

    results: list[dict] = []
    with BiliClient(sessdata=login.sessdata, csrf=login.csrf, uid=login.uid) as client:
        print("\n开始检查（每项 1-2 秒）：\n")
        me = run_check("校验登录态", client.whoami, results)
        if not me["ok"]:
            return finish(results, login.uid)
        print(f"  {' ' * 15}已登录：{login.uname or me['detail']['uname']}（uid {login.uid}）\n")

        folders = run_check("列出收藏夹", lambda: summarise_folders(client), results)

        if folders["ok"] and folders["detail"]["count"]:
            media_id = folders["detail"]["folders"][0]["id"]
            run_check("收藏夹第1页", lambda: summarise_page(client, media_id, 1), results)
            run_check("收藏夹第2页", lambda: summarise_page(client, media_id, 2), results)

    return finish(results, login.uid)


def finish(results: list[dict], uid: int) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"probe-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "uid": uid, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    passed = sum(1 for r in results if r["ok"])
    print("\n" + "-" * 52)
    print(f"结果：{passed}/{len(results)} 项通过")
    print(f"报告：{out}")
    if passed != len(results):
        print(f"日志：{DATA_DIR / 'probe.log'}")
        print("\n有项目失败。把上面带 FAIL 的那几行贴给我。")
        return 1

    print("\n探针跑完了，没有需要你操作的东西。")
    print("下一步是采集层：把收藏夹全量抓下来。")
    return 0


def _pad(text: str, width: int) -> str:
    """Pad by display width: a CJK character occupies two columns."""
    used = sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)
    return text + " " * max(1, width - used)


def run_check(name: str, fn, results: list[dict]) -> dict:
    """Run one check and print the outcome immediately."""
    print(f"  {_pad(name, 14)}", end="", flush=True)
    start = time.perf_counter()
    try:
        detail = fn()
        ok, error = True, None
    except Exception as exc:  # noqa: BLE001 - probe reports everything
        detail, ok, error = None, False, f"{type(exc).__name__}: {exc}"
    ms = round((time.perf_counter() - start) * 1000)

    print(f"{'OK' if ok else 'FAIL'}  {ms:>5}ms", flush=True)
    if ok and detail:
        summary = detail.pop("summary", "") if isinstance(detail, dict) else ""
        if summary:
            print(f"  {' ' * 15}{summary}", flush=True)
    elif error:
        print(f"  {' ' * 15}{error}", flush=True)

    row = {"check": name, "ok": ok, "ms": ms, "detail": detail, "error": error}
    results.append(row)
    return row


def summarise_folders(client: BiliClient) -> dict:
    data = client.folders() or {}
    folders = data.get("list") or []
    biggest = max((f.get("media_count") or 0 for f in folders), default=0)
    return {
        "summary": f"{len(folders)} 个收藏夹，最大 {biggest} 条",
        "count": len(folders),
        "folders": [
            {"id": f.get("id"), "title": f.get("title"), "count": f.get("media_count")}
            for f in folders
        ],
    }


def summarise_page(client: BiliClient, media_id: int, pn: int) -> dict:
    data = client.folder_items(media_id, pn=pn) or {}
    medias = data.get("medias") or []
    first = medias[0] if medias else {}
    missing = [f for f in EXPECTED_ITEM_FIELDS if f not in first]
    return {
        "summary": (
            f"第 {pn} 页返回 {len(medias)} 条，"
            + ("字段齐全" if not missing else f"缺 {', '.join(missing)}")
            + ("，还有下一页" if data.get("has_more") else "，已到末页")
        ),
        "requested_pn": pn,
        "returned": len(medias),
        "first_item_id": first.get("id"),
        "missing_fields": missing,
        "has_more": data.get("has_more"),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotLoggedIn as exc:
        print(f"登录态无效: {exc}")
        raise SystemExit(1) from exc
    except RateLimited as exc:
        print(f"触发风控: {exc}. 停 10 分钟再试。")
        raise SystemExit(1) from exc
    except BiliError as exc:
        print(f"接口报错: {exc}")
        raise SystemExit(1) from exc
