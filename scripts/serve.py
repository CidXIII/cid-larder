#!/usr/bin/env python
"""Start the local web app and open it in the browser."""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402


def free_port(start: int = 8000, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("没有可用端口（8000-8019 都被占用）")


def build_stamp() -> str:
    """The page's mtime, used as a cache key in the URL we open.

    Without this the browser may reuse a cached copy of index.html long after
    it changed, and the page will then disagree with the server it just called.
    Same file, same URL; edited file, new URL, so a stale copy cannot win.
    """
    try:
        return str(int((ROOT / "app" / "web" / "index.html").stat().st_mtime))
    except OSError:
        return "0"


def open_when_ready(url: str, page: str) -> None:
    for _ in range(80):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=0.5)
            break
        except Exception:  # noqa: BLE001 - keep waiting for the server
            time.sleep(0.2)
    webbrowser.open(page)


def main() -> int:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    page = f"{url}/?v={build_stamp()}"

    print("Cid-Larder")
    print(f"  地址：{page}")
    print("  关闭这个窗口即停止")
    print()

    threading.Thread(target=open_when_ready, args=(url, page), daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
