"""Getting a usable session, in one place.

Order: environment variables (development only) -> QR login. Nothing is
written to disk; D15 keeps the session in memory for the life of the process.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters.bili import BiliClient, NotLoggedIn
from ..adapters.bili_login import LoginError, QrState, qr_login
from .config import settings

STATE_TEXT = {
    QrState.WAITING: "等待扫码…",
    QrState.SCANNED: "已扫码，请在手机上确认登录",
    QrState.OK: "登录成功",
    QrState.EXPIRED: "二维码已过期",
}


@dataclass(frozen=True)
class Login:
    sessdata: str
    csrf: str
    uid: int
    uname: str = ""

    def __repr__(self) -> str:  # never print the secrets by accident
        return f"Login(user={self.uname or '?'}, uid={self.uid}, sessdata=***)"


def acquire_session(say=print) -> Login | None:
    """Return a verified login, or None if the user did not complete it."""
    from_env = _from_env()
    if from_env is not None:
        say("使用 .env 里的登录态（开发模式）")
        return _verify(from_env, say)

    say("没有登录态，开始扫码。\n")
    try:
        session = qr_login(
            on_state=lambda state: say(STATE_TEXT.get(state, str(state)))
        )
    except LoginError as exc:
        say(f"登录失败：{exc}")
        return None

    login = Login(
        sessdata=session.sessdata,
        csrf=session.bili_jct,
        uid=int(session.dede_user_id or 0),
    )
    return _verify(login, say)


def _from_env() -> Login | None:
    if not (settings.bili_sessdata and settings.bili_csrf and settings.bili_uid):
        return None
    try:
        uid = int(settings.bili_uid)
    except ValueError:
        return None
    return Login(settings.bili_sessdata, settings.bili_csrf, uid)


def _verify(login: Login, say) -> Login | None:
    """A session is not usable until /x/web-interface/nav accepts it."""
    with BiliClient(sessdata=login.sessdata, csrf=login.csrf) as client:
        try:
            me = client.whoami()
        except NotLoggedIn as exc:
            say("登录态无效，B 站返回未登录。")
            say(f"  {exc}")
            return None

    return Login(
        sessdata=login.sessdata,
        csrf=login.csrf,
        uid=int(login.uid or me["mid"]),
        uname=str(me.get("uname") or ""),
    )
