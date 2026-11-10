"""Model endpoints: show what is configured, test it, save it, list what a key can call.

Model IDs live in providers.json as candidate arrays, never inline in logic:
they change every few months, and a stale name fails the same way a wrong one does.
"""

from __future__ import annotations

import json

import httpx

from .config import CONFIG_PATH, PROVIDERS

# The page calls the second block "emb"; config.json calls it "embedding".
# They drifted once and the symptom was "缺少 API Key", not a naming error.
_BLOCKS = {
    "llm": "llm",
    "chat": "llm",
    "embedding": "embedding",
    "emb": "embedding",
    "vector": "embedding",
}


def _block(name: str) -> str | None:
    """Resolve either spelling to the name config.json uses."""
    return _BLOCKS.get((name or "").strip().lower())


def _candidates(provider: str, block: str) -> list[str]:
    preset = PROVIDERS.get(provider, {})
    field = "chat_models" if block == "llm" else "embedding_models"
    values = preset.get(field) or []
    return [str(item) for item in values if item]


def _default_model(provider: str, block: str) -> str:
    """First candidate, or empty when we have none. Never guesses."""
    candidates = _candidates(provider, block)
    return candidates[0] if candidates else ""


def presets() -> list[dict]:
    return [
        {
            "id": key,
            "label": value.get("label", key),
            "base_url": value.get("base_url", ""),
            "chat_models": _candidates(key, "llm"),
            "embedding_models": _candidates(key, "embedding"),
            "note": value.get("note", ""),
            "source": value.get("source", ""),
        }
        for key, value in sorted(PROVIDERS.items())
    ]


def read() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(payload: dict) -> None:
    """Merge one or both endpoint blocks into config.json."""
    config = read()
    for raw_block, data in payload.items():
        block = _block(raw_block)
        if block is None or not isinstance(data, dict):
            continue
        provider = (data.get("provider") or "").strip()
        if not provider:
            continue
        preset = PROVIDERS.get(provider, {})
        config[f"{block}_provider"] = provider
        if data.get("api_key"):
            config[f"{block}_api_key"] = data["api_key"].strip()
        config[f"{block}_model"] = (
            (data.get("model") or "").strip() or _default_model(provider, block)
        )
        config[f"{block}_base_url"] = (data.get("base_url") or "").strip() or preset.get("base_url", "")

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def current() -> dict:
    config = read()
    out: dict = {}
    for block in ("llm", "embedding"):
        provider = config.get(f"{block}_provider") or ""
        preset = PROVIDERS.get(provider, {})
        key = config.get(f"{block}_api_key") or ""
        out[block] = {
            "provider": provider,
            "label": preset.get("label", provider),
            "model": config.get(f"{block}_model") or _default_model(provider, block),
            "base_url": config.get(f"{block}_base_url") or preset.get("base_url", ""),
            "has_key": bool(key),
            "masked": _mask(key),
        }
    out["ready"] = bool(out["llm"]["has_key"] and out["embedding"]["has_key"])
    return out


def endpoint(block: str = "llm") -> dict | None:
    """Resolved endpoint with the raw key, or None when not configured."""
    block = _block(block) or ""
    config = read()
    provider = config.get(f"{block}_provider") or ""
    key = config.get(f"{block}_api_key") or ""
    if not provider or not key:
        return None
    preset = PROVIDERS.get(provider, {})
    return {
        "provider": provider,
        "base_url": config.get(f"{block}_base_url") or preset.get("base_url", ""),
        "model": config.get(f"{block}_model") or _default_model(provider, block),
        "api_key": key,
    }


def list_models(
    block: str, provider: str, api_key: str, base_url: str = ""
) -> tuple[bool, list[str] | str]:
    """Ask the endpoint which model IDs this key can call.

    Uses the OpenAI-compatible GET /models route. Not every host implements it,
    which is a normal answer, not a failure: the user can still type a name.
    """
    block = _block(block)
    if block is None:
        return False, "未知的模型类型"
    preset = PROVIDERS.get(provider)
    if preset is None:
        return False, f"未知的 provider：{provider}"

    base = (base_url or preset.get("base_url") or "").rstrip("/")
    # An empty box means "use the saved key", so fetching works without
    # re-pasting a key the user already stored.
    api_key = api_key or (read().get(f"{block}_api_key") or "")
    if not base:
        return False, "缺少 base_url"
    if not api_key:
        return False, "缺少 API Key"

    try:
        response = httpx.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20.0,
        )
    except httpx.HTTPError as exc:
        return False, f"连不上：{type(exc).__name__}"

    if response.status_code == 401:
        return False, "Key 无效（401）"
    if response.status_code == 404:
        return False, "这个服务商没有 /models 接口，直接手填模型名"
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}{_error_text(response)}"

    try:
        payload = response.json()
    except ValueError:
        return False, "返回的不是 JSON"

    models = _extract_ids(payload)
    if not models:
        return False, "接口通了，但响应里没有模型列表"
    return True, models


def _extract_ids(payload: object) -> list[str]:
    """OpenAI-style {"data":[{"id":...}]}, with fallbacks for other shapes."""
    if not isinstance(payload, dict):
        return []
    found: list[str] = []
    for field in ("data", "models"):
        entries = payload.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("id") or entry.get("name") or entry.get("model")
                if name:
                    found.append(str(name))
            elif isinstance(entry, str):
                found.append(entry)
        if found:
            break
    return sorted(set(found), key=str.lower)


def probe(block: str, provider: str, api_key: str, base_url: str = "", model: str = "") -> tuple[bool, str]:
    """Send the smallest possible request to prove the endpoint works."""
    block = _block(block)
    if block is None:
        return False, "未知的模型类型"
    preset = PROVIDERS.get(provider)
    if preset is None:
        return False, f"未知的 provider：{provider}"

    base = (base_url or preset.get("base_url") or "").rstrip("/")
    name = model or _default_model(provider, block)
    # Same rule as list_models: an empty box means the saved key.
    api_key = api_key or (read().get(f"{block}_api_key") or "")
    if not base:
        return False, "缺少 base_url"
    if not name:
        return False, "缺少模型名"
    if not api_key:
        return False, "缺少 API Key"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if block == "llm":
        url, body = f"{base}/chat/completions", {
            "model": name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
    else:
        url, body = f"{base}/embeddings", {"model": name, "input": "ping"}

    try:
        response = httpx.post(url, json=body, headers=headers, timeout=25.0)
    except httpx.HTTPError as exc:
        return False, f"连不上：{type(exc).__name__}"

    if response.status_code == 200:
        return True, "连接正常"

    detail = _error_text(response)
    if response.status_code == 401:
        return False, f"Key 无效（401）{detail}"
    if response.status_code == 403:
        return False, f"没有权限（403）{detail}"
    if response.status_code == 404:
        return False, f"地址或模型名不对（404）{detail}"
    if response.status_code == 429:
        return False, f"被限流（429），稍后再试{detail}"
    return False, f"HTTP {response.status_code}{detail}"


def _error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return f"：{str(message)[:120]}"
    return ""


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}{'•' * 6}{key[-4:]}"
