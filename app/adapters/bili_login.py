"""Bilibili QR login.

Generate a key, poll until the phone confirms, then read the cookies. The
session lives in memory only -- D15.

Where the cookies come from depends on the flow the server picks that day:
older responses embed SESSDATA in the callback URL, newer ones hand back a
ticket and set the cookies via Set-Cookie. Both are handled, and the one that
worked is logged by cookie NAME only, never by value.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from urllib.parse import urlparse

import httpx

PASSPORT = "https://passport.bilibili.com"
QR_GENERATE = "/x/passport-login/web/qrcode/generate"
QR_POLL = "/x/passport-login/web/qrcode/poll"

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class QrState(IntEnum):
    OK = 0
    WAITING = 86101
    SCANNED = 86090
    EXPIRED = 86038


@dataclass(frozen=True)
class Session:
    sessdata: str
    bili_jct: str
    dede_user_id: str = ""

    def __repr__(self) -> str:  # never print the secrets by accident
        return f"Session(user={self.dede_user_id or '?'}, sessdata=***)"


class LoginError(RuntimeError):
    pass


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=PASSPORT,
        timeout=10.0,
        headers={"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
    )


def generate_qr() -> tuple[str, str]:
    """Return (qrcode_key, qr_content_url)."""
    with _client() as client:
        payload = client.get(QR_GENERATE).json()
    if payload.get("code") != 0:
        raise LoginError(f"生成二维码失败: {payload.get('code')} {payload.get('message')}")
    data = payload.get("data") or {}
    key, url = data.get("qrcode_key"), data.get("url")
    if not key or not url:
        raise LoginError("二维码响应里缺少 qrcode_key 或 url")
    log.debug("qr generated: key_len=%d url_len=%d", len(key), len(url))
    return key, url


def _poll(client: httpx.Client, qrcode_key: str) -> tuple[QrState, Session | None]:
    response = client.get(QR_POLL, params={"qrcode_key": qrcode_key})
    payload = response.json()

    if payload.get("code") != 0:
        raise LoginError(f"轮询失败: {payload.get('code')} {payload.get('message')}")

    data = payload.get("data") or {}
    raw_code = data.get("code", -1)
    state = QrState(raw_code) if raw_code in QrState._value2member_map_ else QrState.EXPIRED
    if state is not QrState.OK:
        return state, None

    callback = data.get("url") or ""
    params = _parse_callback(callback)
    log.info(
        "login confirmed: data_keys=%s callback=%s%s keys=%s set_cookie=%s",
        sorted(data),
        urlparse(callback).hostname or "-",
        urlparse(callback).path or "-",
        sorted(params),
        sorted(_cookies_from_headers(response)),
    )

    jar = _collect_cookies(client, response, callback, params)
    sessdata = jar.get("SESSDATA", "")
    bili_jct = jar.get("bili_jct", "")
    dede_user_id = jar.get("DedeUserID", "")

    if not (sessdata and bili_jct):
        raise LoginError(
            "扫码成功，但没拿到 SESSDATA / bili_jct。"
            f"回调 {sorted(params)}，最终拿到的 cookie {sorted(jar)}"
        )
    log.info(
        "session ok: sessdata_len=%d jct_len=%d has_user_id=%s",
        len(sessdata), len(bili_jct), bool(dede_user_id),
    )
    return state, Session(sessdata, bili_jct, dede_user_id)


def _collect_cookies(
    client: httpx.Client,
    response: httpx.Response,
    callback: str,
    params: dict[str, str],
) -> dict[str, str]:
    """Gather cookies from every place the server might put them.

    Current flow: the poll hands back a cross-domain `ticket`, and only the
    follow-up request sets SESSDATA / bili_jct in Set-Cookie. Older responses
    carried them in the callback query. Try the query, the poll response, the
    callback itself, then the known exchange endpoints.
    """
    # Old flow put credentials in the query. Keep only the three we need so
    # ticket / gourl do not pollute the jar.
    jar: dict[str, str] = {
        k: params[k] for k in ("SESSDATA", "bili_jct", "DedeUserID") if k in params
    }
    jar.update(_cookies_from_headers(response))

    if _has_credentials(jar):
        return jar

    if callback:
        _merge(jar, _from_chain(_get(client, callback)), "callback")

    if _has_credentials(jar):
        return jar

    ticket = params.get("ticket", "")
    if ticket:
        gourl = params.get("gourl", "")
        for url in (
            f"https://passport.biligame.com/x/passport-login/web/crossDomain?ticket={ticket}&gourl={gourl}",
            f"https://passport.bilibili.com/x/passport-login/web/crossDomain?ticket={ticket}&gourl={gourl}",
            f"https://passport.biligame.com/crossDomain?ticket={ticket}&gourl={gourl}",
        ):
            _merge(jar, _from_chain(_get(client, url)), urlparse(url).netloc)
            if _has_credentials(jar):
                break

    return jar


def _has_credentials(jar: dict[str, str]) -> bool:
    return bool(jar.get("SESSDATA") and jar.get("bili_jct"))


def _get(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        return client.get(url, follow_redirects=True)
    except httpx.HTTPError as exc:
        log.info("callback fetch failed: %s %s", urlparse(url).netloc, type(exc).__name__)
        return None


def _from_chain(response: httpx.Response | None) -> dict[str, str]:
    """Cookies sit on whichever hop of the redirect chain set them."""
    if response is None:
        return {}
    jar = _cookies_from_headers(response)
    log.info(
        "callback chain: hops=%d final=%s cookies=%s",
        len(response.history), response.status_code, sorted(jar),
    )
    return jar


def _cookies_from_headers(response: httpx.Response) -> dict[str, str]:
    """Read Set-Cookie straight off the headers.

    A cookie jar would drop anything whose `Domain` does not match the host
    that sent it -- and the ticket exchange sets `.bilibili.com` cookies from
    a `passport.biligame.com` response. We only need name/value pairs to build
    our own Cookie header, so domain rules are irrelevant here.
    """
    jar: dict[str, str] = {}
    for hop in (*response.history, response):
        for raw in hop.headers.get_list("set-cookie"):
            pair = raw.split(";", 1)[0]
            name, sep, value = pair.partition("=")
            if sep and name.strip() and value.strip():
                jar[name.strip()] = value.strip()
    return jar


def _merge(jar: dict[str, str], extra: dict[str, str], source: str) -> None:
    """Later reads win: a live Set-Cookie beats anything parsed from the URL."""
    changed = [k for k, v in extra.items() if jar.get(k) != v]
    if changed:
        log.info("cookies from %s: %s", source, sorted(changed))
    jar.update(extra)


def _parse_callback(callback: str) -> dict[str, str]:
    """Parse the query WITHOUT percent-decoding.

    SESSDATA contains %2C literally and the browser sends it that way.
    parse_qs would turn %2C into ',', which is a different cookie value and
    gets rejected as not-logged-in.
    """
    query = urlparse(callback).query
    result: dict[str, str] = {}
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        result[key] = value
    return result


def _render(qr, url: str) -> None:
    """ASCII in the terminal; SVG file if the console cannot encode it."""
    try:
        qr.print_ascii(invert=True)
        sys.stdout.flush()
        print("\n用 B 站 App 扫上面的二维码。", flush=True)
        return
    except UnicodeEncodeError:
        pass

    from pathlib import Path

    out = Path.cwd() / "data" / "login-qr.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    import qrcode.image.svg

    qr2 = qrcode.QRCode(border=2, image_factory=qrcode.image.svg.SvgPathImage)
    qr2.add_data(url)
    qr2.make(fit=True)
    qr2.make_image().save(str(out))
    print(f"终端显示不了二维码，已存成图片：{out}")
    print("用浏览器打开它，再用 B 站 App 扫。")


def qr_login(on_state=None, interval: float = 2.0, timeout: float = 180.0) -> Session:
    """Block until the user scans, or raise on timeout/expiry."""
    import qrcode

    key, url = generate_qr()
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    _render(qr, url)

    deadline = time.monotonic() + timeout
    last = None
    # One client for the whole wait: fewer TLS handshakes, and the cookies the
    # server sets during the ticket exchange stay in the same session.
    with _client() as client:
        while time.monotonic() < deadline:
            state, session = _poll(client, key)
            if state is QrState.OK and session is not None:
                if on_state:
                    on_state(state)
                return session
            if state is not last:
                log.debug("poll state=%s", state.name)
                if on_state:
                    on_state(state)
                last = state
            if state is QrState.EXPIRED:
                raise LoginError("二维码已过期，请重新运行")
            time.sleep(interval)

    raise LoginError("等待扫码超时")


# --- web flow -------------------------------------------------------------
# The terminal flow blocks; a browser needs to poll instead. One httpx client
# is kept per pending QR so the ticket exchange stays in the same session.

_pending: dict[str, tuple[httpx.Client, float]] = {}
_pending_lock = threading.Lock()
PENDING_TTL = 300.0


def start_login() -> tuple[str, str]:
    """Return (qrcode_key, qr_content_url) and keep a client for polling."""
    _sweep_pending()
    key, url = generate_qr()
    with _pending_lock:
        _pending[key] = (_client(), time.monotonic())
    return key, url


def poll_login(key: str) -> tuple[QrState, Session | None]:
    with _pending_lock:
        entry = _pending.get(key)
    if entry is None:
        entry = (_client(), time.monotonic())
        with _pending_lock:
            _pending[key] = entry

    client = entry[0]
    state, session = _poll(client, key)

    # Scanned-and-confirmed or expired: this key is done either way.
    if state in (QrState.OK, QrState.EXPIRED):
        with _pending_lock:
            _pending.pop(key, None)
        client.close()
    return state, session


def _sweep_pending() -> None:
    now = time.monotonic()
    with _pending_lock:
        stale = [k for k, (_, born) in _pending.items() if now - born > PENDING_TTL]
        clients = [_pending.pop(k)[0] for k in stale]
    for client in clients:
        client.close()


def qr_svg(url: str) -> str:
    """Inline SVG for the browser: qrcode draws paths, so no raster image."""
    import qrcode
    import qrcode.image.svg

    image = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")
