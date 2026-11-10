"""Bilibili web API client. Read-only endpoints only, for now.

Error handling is centralised in `_request`: every endpoint returns `code: 0`
on success, so anything else is mapped to a typed exception here rather than
being checked at each call site.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field

import httpx

API_ROOT = "https://api.bilibili.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

log = logging.getLogger(__name__)

# Endpoint maxima, not tunables. Never derive a total from page size: a page
# can come back short without being the last one. Paginate on `has_more`.
FAV_PAGE_SIZE = 20
# Ask for more than we expect and let the server tell us its real cap: the
# count it returns while `has_more` is true IS the page size.
PROBE_PAGE_SIZE = 100

NOT_LOGGED_IN = {-101}
RATE_LIMITED = {-352, -412, -799}
# Bilibili answers risk control with a bare HTTP status often enough to matter.
HTTP_RATE_LIMITED = {412, 429}

# folder/add treats an omitted privacy flag as public.
PRIVACY_PRIVATE = 1

# The move endpoint takes a comma-separated list but documents no cap. Twenty
# keeps one bad request from taking a large slice with it.
BATCH_MOVE = 20


class BiliError(RuntimeError):
    def __init__(self, code: int, message: str, path: str) -> None:
        super().__init__(f"[{code}] {message} ({path})")
        self.code = code
        self.message = message
        self.path = path


class NotLoggedIn(BiliError):
    """Session missing or expired. The UI should send the user back to login."""


class RateLimited(BiliError):
    """Risk control triggered. Stop the batch; do not retry immediately."""


@dataclass
class BiliClient:
    sessdata: str = ""
    csrf: str = ""
    uid: int | None = None
    min_interval: float = 1.2
    jitter: float = 0.8
    timeout: float = 10.0
    _last_call: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            base_url=API_ROOT,
            timeout=self.timeout,
            headers={
                "User-Agent": USER_AGENT,
                "Referer": "https://www.bilibili.com/",
            },
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BiliClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _cookie_header(self) -> str:
        parts = []
        if self.sessdata:
            parts.append(f"SESSDATA={self.sessdata}")
        if self.csrf:
            parts.append(f"bili_jct={self.csrf}")
        return "; ".join(parts)

    def _throttle(self) -> None:
        """Reserve the next departure slot, then sleep until it.

        The lock only covers the reservation, not the request itself, so a few
        requests can be in flight while their start times stay spaced out.
        """
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval + random.uniform(0, self.jitter) - (now - self._last_call)
            wait = max(wait, 0.0)
            self._last_call = now + wait
        if wait:
            time.sleep(wait)

    def _request(self, method: str, path: str, **kwargs: object) -> object:
        self._throttle()
        headers = {}
        cookie = self._cookie_header()
        if cookie:
            headers["Cookie"] = cookie
        response = self._http.request(method, path, headers=headers, **kwargs)
        status = response.status_code

        try:
            payload = response.json()
        except ValueError:
            payload = None

        # Risk control and gateway errors often come back as HTML, not JSON.
        if payload is None:
            snippet = " ".join(response.text.split())[:120]
            if status in HTTP_RATE_LIMITED:
                raise RateLimited(-status, snippet or f"HTTP {status}", path)
            raise BiliError(-status, snippet or f"HTTP {status}", path)

        code = int(payload.get("code", 0))
        log.debug("%s %s -> status=%s code=%s", method, path, status, code)
        if code == 0 and status < 400:
            return payload.get("data")

        message = str(payload.get("message") or "") or f"HTTP {status}"
        if status in HTTP_RATE_LIMITED or code in RATE_LIMITED:
            raise RateLimited(code or -status, message, path)
        if code in NOT_LOGGED_IN:
            raise NotLoggedIn(code, message, path)
        raise BiliError(code or -status, message, path)

    def get(self, path: str, params: dict | None = None) -> object:
        return self._request("GET", path, params=params)

    def post(self, path: str, data: dict | None = None) -> object:
        body = dict(data or {})
        body.setdefault("csrf", self.csrf)
        body.setdefault("platform", "web")
        return self._request("POST", path, data=body)

    def folders(self) -> dict:
        if not self.uid:
            raise ValueError("uid is required to list folders")
        return self.get("/x/v3/fav/folder/created/list-all", {"up_mid": self.uid})  # type: ignore[return-value]

    def folder_items(self, media_id: int, pn: int = 1, ps: int = FAV_PAGE_SIZE) -> dict:
        return self.get(  # type: ignore[return-value]
            "/x/v3/fav/resource/list",
            {"media_id": media_id, "pn": pn, "ps": ps, "platform": "web"},
        )

    def whoami(self) -> dict:
        """Verify the session and return mid / uname / isLogin."""
        data = self.get("/x/web-interface/nav") or {}
        if not data.get("isLogin"):
            raise NotLoggedIn(
                int(data.get("code", -101)),
                f"isLogin=false (code={data.get('code')})",
                "/x/web-interface/nav",
            )
        return {
            "mid": data.get("mid"),
            "uname": data.get("uname"),
            "level": (data.get("level_info") or {}).get("current_level"),
        }

    def folder_ids(self, media_id: int) -> list:
        """Every item id in a folder, no pagination. Ids only, no metadata."""
        data = self.get(  # type: ignore[return-value]
            "/x/v3/fav/resource/ids",
            {"media_id": media_id, "platform": "web"},
        )
        return data or []

    def iter_folder_items(self, media_id: int, ps: int = FAV_PAGE_SIZE):
        """Yield every item in a folder, following `has_more`.

        Stops on `has_more == false`, not on a page count, and bails out on an
        empty page so a misbehaving endpoint cannot loop forever.
        """
        pn = 1
        while True:
            data = self.folder_items(media_id, pn=pn, ps=ps) or {}
            medias = data.get("medias") or []
            if not medias:
                return
            yield from medias
            if not data.get("has_more"):
                return
            pn += 1

    def create_folder(self, title: str, intro: str = "") -> int:
        """Create a private folder and return its media_id.

        `privacy` defaults to public on this endpoint, so it is always sent
        explicitly. A category folder must never be public by accident.
        """
        data = self.post(
            "/x/v3/fav/folder/add",
            {"title": title, "intro": intro, "privacy": PRIVACY_PRIVATE},
        )
        try:
            return int(data["id"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise BiliError(-1, "创建收藏夹后没有拿到 id", "/x/v3/fav/folder/add") from exc

    def delete_folder(self, media_id: int) -> None:
        """Delete one of your own folders.

        The endpoint takes a list; one id is still a list. It refuses the
        default folder and anything that still holds a video, which is the
        behaviour we want anyway - callers check first so a refusal is rare.
        """
        self.post("/x/v3/fav/folder/del", {"media_ids": str(media_id)})

    def move_items(self, src_media_id: int, tar_media_id: int, aids: list[int]) -> None:
        """Move videos between two of your own folders."""
        if not self.uid:
            raise ValueError("uid is required to move items")
        if not aids:
            return
        resources = ",".join(f"{aid}:2" for aid in aids)
        self.post(
            "/x/v3/fav/resource/move",
            {
                "src_media_id": src_media_id,
                "tar_media_id": tar_media_id,
                "mid": self.uid,
                "resources": resources,
            },
        )
