"""In-memory state for the local web app.

The platform session lives here and nowhere else (D15). Restarting the process
means scanning again; that is the intended trade.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..adapters.bili import BiliClient, BiliError, NotLoggedIn, RateLimited
from .collector import Summary, collect
from .execute import ExecuteError, Interrupted, run as run_execute
from .db import connect, init_db


@dataclass(frozen=True)
class LoginInfo:
    sessdata: str
    csrf: str
    uid: int
    uname: str

    def public(self) -> dict:
        return {"logged_in": True, "uid": self.uid, "uname": self.uname}


class Runtime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.login: LoginInfo | None = None
        self._thread: threading.Thread | None = None
        self.progress: dict = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {
            "running": False,
            "finished": False,
            "stage": "",
            "done": 0,
            "total": 0,
            "note": "",
            "summary": {},
            "error": "",
            "started_at": 0.0,
        }

    # --- login -----------------------------------------------------------

    def set_login(self, sessdata: str, csrf: str, uid: int, uname: str) -> LoginInfo:
        self.login = LoginInfo(sessdata, csrf, uid, uname)
        return self.login

    def clear_login(self) -> None:
        self.login = None

    # --- collect ---------------------------------------------------------

    def collect_status(self) -> dict:
        with self._lock:
            return dict(self.progress)

    def start_collect(self) -> tuple[bool, str]:
        login = self.login
        if login is None:
            return False, "未登录"
        with self._lock:
            if self.progress["running"]:
                return False, "已经在采集了"
            self.progress = self._idle()
            self.progress.update(running=True, started_at=time.time())

        self._thread = threading.Thread(target=self._run, args=(login,), daemon=True)
        self._thread.start()
        return True, ""

    def _run(self, login: LoginInfo) -> None:
        conn = connect()
        try:
            init_db(conn)
            with BiliClient(
                sessdata=login.sessdata,
                csrf=login.csrf,
                uid=login.uid,
                min_interval=0.35,
                jitter=0.2,
            ) as client:
                summary: Summary = collect(client, conn, progress=self._progress)
            self._finish(summary=summary.as_dict())
        except NotLoggedIn:
            self.clear_login()
            self._finish(error="登录已失效，请重新扫码")
        except RateLimited as exc:
            self._finish(error=f"触发风控，停几分钟再试：{exc}")
        except BiliError as exc:
            self._finish(error=f"接口报错：{exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self._finish(error=f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()

    # --- execute ---------------------------------------------------------

    def start_execute(self) -> tuple[bool, str]:
        login = self.login
        if login is None:
            return False, "未登录"
        with self._lock:
            if self.progress["running"]:
                return False, "有任务正在跑，等它结束"
            self.progress = self._idle()
            self.progress.update(running=True, started_at=time.time())

        threading.Thread(target=self._run_execute, args=(login,), daemon=True).start()
        return True, ""

    def _run_execute(self, login: LoginInfo) -> None:
        conn = connect()
        try:
            init_db(conn)
            # Slower than collection on purpose: these requests change state.
            with BiliClient(
                sessdata=login.sessdata,
                csrf=login.csrf,
                uid=login.uid,
                min_interval=0.6,
                jitter=0.3,
            ) as client:
                result = run_execute(conn, client, progress=self._progress)
            self._finish(summary=result)
        except Interrupted as exc:
            if exc.login_lost:
                self.clear_login()
            # Both halves matter: why it stopped, and how much of it landed.
            self._finish(summary=exc.as_summary(), error=exc.reason)
        except NotLoggedIn:
            self.clear_login()
            self._finish(error="登录已失效，已停止；已经移动的条目仍在目标收藏夹里")
        except RateLimited as exc:
            self._finish(error=f"触发风控，已停止。已经移动的部分不回滚：{exc}")
        except ExecuteError as exc:
            self._finish(error=str(exc))
        except BiliError as exc:
            self._finish(error=f"接口报错，已停止：{exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced to the page
            self._finish(error=f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()

    def _progress(self, stage: str, done: int, total: int, note: str) -> None:
        with self._lock:
            self.progress.update(stage=stage, done=done, total=total, note=note)

    def _finish(self, summary: dict | None = None, error: str = "") -> None:
        with self._lock:
            self.progress.update(
                running=False, finished=True, summary=summary or {}, error=error
            )


runtime = Runtime()
