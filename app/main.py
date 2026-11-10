"""Local web app. Binds to 127.0.0.1 only; nothing is exposed to the network."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .adapters.bili import BiliClient, BiliError, NotLoggedIn
from .adapters.bili_login import QrState, poll_login, qr_svg, start_login
from .core.db import connect, init_db
from .core import backup, classify, models, store, taxonomy, usage
from .core.config import reload_settings
from .core.library import items_in, overview
from .core.runtime import runtime

WEB_DIR = Path(__file__).parent / "web"

# Bump when a response shape changes. The page compares it and tells the user
# to restart, instead of silently rendering blanks against an old process.
API_VERSION = 57

STATE_NAMES = {
    QrState.WAITING: "waiting",
    QrState.SCANNED: "scanned",
    QrState.OK: "ok",
    QrState.EXPIRED: "expired",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # In-memory database: starts empty, and nothing survives the process.
    init_db()
    yield


app = FastAPI(title="Cid-Larder", docs_url="/api/docs", lifespan=lifespan)


@app.middleware("http")
async def guard_writes(request: Request, call_next):
    """A custom header makes cross-site form posts fail before they arrive.

    Any page in the browser can POST to 127.0.0.1; requiring a header forces a
    preflight that a hostile page cannot satisfy.
    """
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if request.headers.get("x-cid-larder") != "1":
            return JSONResponse({"error": "missing client header"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
            return JSONResponse({"error": "bad origin"}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def no_store(request: Request, call_next):
    """A local app has no cacheable state; a cached reply is only ever wrong."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def build_stamp() -> str:
    """The page's mtime. It changes whenever the page does, which is the only
    thing that matters for deciding whether a cached copy is out of date."""
    try:
        return str(int((WEB_DIR / "index.html").stat().st_mtime))
    except OSError:
        return "0"


@app.get("/")
def index(request: Request):
    # A bare address may still sit in the browser cache from before no-store
    # existed. Send it to a URL that carries the build it should be showing.
    if "v" not in request.query_params:
        return RedirectResponse(f"/?v={build_stamp()}", status_code=302)
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/state")
def state() -> dict:
    login = runtime.login
    payload = login.public() if login else {"logged_in": False}
    return {**payload, "api": API_VERSION}


@app.post("/api/login/start")
def login_start() -> dict:
    try:
        key, url = start_login()
    except Exception as exc:  # noqa: BLE001 - surfaced to the page
        raise HTTPException(status_code=502, detail=f"生成二维码失败：{exc}") from exc
    return {"key": key, "svg": qr_svg(url)}


@app.get("/api/login/poll")
def login_poll(key: str) -> dict:
    try:
        qr_state, session = poll_login(key)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"轮询失败：{exc}") from exc

    name = STATE_NAMES.get(qr_state, "expired")
    if name != "ok" or session is None:
        return {"state": name}

    # A session is not usable until /nav accepts it, so verify here rather than
    # discovering it during the first collection request.
    with BiliClient(sessdata=session.sessdata, csrf=session.bili_jct) as client:
        try:
            me = client.whoami()
        except NotLoggedIn:
            return {"state": "expired", "detail": "登录态未通过校验"}

    runtime.set_login(
        session.sessdata, session.bili_jct, int(me["mid"]), str(me.get("uname") or "")
    )
    return {"state": "ok", "uid": me["mid"], "uname": me.get("uname")}


@app.post("/api/logout")
def logout() -> dict:
    runtime.clear_login()
    return {"ok": True}


@app.post("/api/collect/start")
def collect_start() -> dict:
    ok, reason = runtime.start_collect()
    if not ok:
        raise HTTPException(status_code=409, detail=reason)
    return {"ok": True}


@app.get("/api/collect/status")
def collect_status() -> dict:
    return runtime.collect_status()


class ProbeBody(BaseModel):
    block: str
    provider: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class SaveBody(BaseModel):
    llm: dict | None = None
    embedding: dict | None = None


@app.get("/api/models")
def models_state() -> dict:
    return {"presets": models.presets(), "current": models.current()}


@app.post("/api/models/test")
def models_test(body: ProbeBody) -> dict:
    ok, message = models.probe(body.block, body.provider, body.api_key, body.base_url, body.model)
    return {"ok": ok, "message": message}


@app.post("/api/models/list")
def models_list(body: ProbeBody) -> dict:
    """Ask the provider which model IDs this key can call."""
    ok, result = models.list_models(body.block, body.provider, body.api_key, body.base_url)
    if not ok:
        return {"ok": False, "message": result}
    return {"ok": True, "models": result}


@app.post("/api/models/save")
def models_save(body: SaveBody) -> dict:
    models.save(body.model_dump(exclude_none=True))
    reload_settings()
    return {"ok": True, "current": models.current()}


class SaveTaxonomyBody(BaseModel):
    items: list[dict]


@app.get("/api/taxonomy")
def taxonomy_state() -> dict:
    conn = connect()
    try:
        return {"items": taxonomy.load(conn)}
    finally:
        conn.close()


class SuggestBody(BaseModel):
    # How many groups to aim for. The page owns the number.
    count: int | None = None


@app.post("/api/taxonomy/suggest")
def taxonomy_suggest(body: SuggestBody | None = None) -> dict:
    conn = connect()
    try:
        items, used = taxonomy.suggest(
            conn, count=body.count if body else None
        )
        # The suggestion goes straight into the store: the page no longer has a
        # save button, and a taxonomy that only exists in the browser is exactly
        # the kind of half-state that made the buttons confusing.
        taxonomy.save(conn, items)
        return {"items": taxonomy.load(conn) or items, "usage": used}
    except taxonomy.TaxonomyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/taxonomy/save")
def taxonomy_save(body: SaveTaxonomyBody) -> dict:
    conn = connect()
    try:
        taxonomy.save(conn, body.items)
        # Anything that used to point at a category that is now gone ends up in
        # 存疑 rather than disappearing from the page.
        dropped = classify.retarget_orphans(
            conn, [taxonomy.display_name(item) for item in body.items]
        )
        return {"ok": True, "items": taxonomy.load(conn), "dropped": dropped}
    except taxonomy.TaxonomyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/api/classify")
def classify_run() -> dict:
    conn = connect()
    try:
        return classify.run(conn)
    except classify.ClassifyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/plan")
def plan_state() -> dict:
    conn = connect()
    try:
        return {"plan": classify.plan(conn)}
    finally:
        conn.close()


@app.get("/api/backup")
def backup_export() -> dict:
    """Snapshot of the current library, for the browser to save.

    The server writes nothing: the response goes out as a download.
    """
    conn = connect()
    try:
        return backup.dump(conn)
    finally:
        conn.close()


@app.post("/api/execute")
def execute_plan() -> dict:
    """Start moving items into their planned folders, in the background."""
    ok, message = runtime.start_execute()
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"ok": True}


class RemoveFoldersBody(BaseModel):
    ids: list[int] = []


@app.post("/api/folders/remove")
def folders_remove(body: RemoveFoldersBody) -> dict:
    """Delete folders a run emptied, after the page has listed them.

    The client's word is not enough: every id is checked against the account
    as it stands now - it must still exist and still be empty. Which folders
    to delete is the person's call; the page lists only the ones this run
    emptied, and asks before anything is removed.
    """
    login = runtime.login
    if login is None:
        raise HTTPException(status_code=409, detail="未登录")

    conn = connect()
    try:
        with BiliClient(
            sessdata=login.sessdata,
            csrf=login.csrf,
            uid=login.uid,
            min_interval=0.6,
            jitter=0.3,
        ) as client:
            data = client.folders() or {}
            by_id: dict[int, dict] = {}
            for folder in data.get("list") or []:
                try:
                    by_id[int(folder.get("id"))] = folder
                except (TypeError, ValueError):
                    continue

            removed: list[dict] = []
            skipped: list[dict] = []
            for folder_id in body.ids:
                folder = by_id.get(folder_id)
                title = (folder.get("title") or "").strip() if folder else ""
                if folder is None:
                    skipped.append({"id": folder_id, "reason": "不在收藏夹列表里"})
                    continue
                if folder_id == 0:
                    skipped.append({"id": folder_id, "name": title, "reason": "默认收藏夹不能删"})
                    continue
                if folder.get("media_count") != 0:
                    skipped.append({"id": folder_id, "name": title, "reason": "里面还有内容"})
                    continue
                try:
                    client.delete_folder(folder_id)
                except BiliError as exc:
                    skipped.append({"id": folder_id, "name": title, "reason": str(exc)})
                    continue
                store.forget_container(conn, "bilibili", "favorites", str(folder_id))
                removed.append({"id": folder_id, "name": title})
            return {"removed": removed, "skipped": skipped}
    finally:
        conn.close()


@app.get("/api/usage")
def usage_state() -> dict:
    conn = connect()
    try:
        return usage.summary(conn)
    finally:
        conn.close()


@app.get("/api/library/container/{container_id}")
def library_container(container_id: str) -> dict:
    conn = connect()
    try:
        return items_in(conn, container_id)
    finally:
        conn.close()


@app.get("/api/plan/items")
def plan_items(target: str) -> dict:
    conn = connect()
    try:
        return classify.items_of(conn, target)
    finally:
        conn.close()


@app.get("/api/library")
def library() -> dict:
    conn = connect()
    try:
        return overview(conn)
    finally:
        conn.close()


class KeepOutBody(BaseModel):
    excluded: bool = True


@app.post("/api/library/container/{container_id}/exclude")
def keep_out(container_id: str, body: KeepOutBody) -> dict:
    """Leave one folder out of the next run, or put it back.

    Nothing on Bilibili is touched. Excluding also drops the folder's items
    from the current plan, because a plan that still queues them for a move
    would contradict the exclusion.
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT name FROM container WHERE platform = 'bilibili' "
            "AND kind = 'favorites' AND container_id = ?",
            (container_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="没有这个收藏夹，先采集一次")
        if (row["name"] or "") == classify.PARKING_NAME:
            raise HTTPException(
                status_code=400,
                detail="「存疑」是停放处，不能剔除；不想要它就让分类别再往里塞",
            )

        changed = store.set_keep_out(conn, container_id, body.excluded)
        dropped = classify.drop_container(conn, container_id) if body.excluded else 0
        return {
            "ok": True,
            "excluded": body.excluded,
            "changed": changed,
            "removed": dropped,
            "plan": classify.plan(conn),
        }
    finally:
        conn.close()
