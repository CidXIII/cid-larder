"""Category suggestions: the model reads the whole library and proposes names.

It only proposes. Nothing is assigned to an item here; that is the next step.

Two failure modes cost us a round trip each, so both are handled here rather
than left to the caller:

- a model that thinks by default can spend the whole token budget on reasoning
  and return an empty `content`;
- a model that ignores "only JSON" and wraps the answer in prose or a fence.

Neither is an exception the user can act on, so both are turned into a message
that says what actually happened.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

import httpx

from . import itemtext, models, platforms, usage

# How many groups the page asks for. The number is the person's call, not the
# prompt's: this is the fallback when the request does not carry one, plus the
# range it is clamped to.
DEFAULT_GROUPS = 15
MIN_GROUPS = 1
MAX_GROUPS = 50

PROMPT = """你在帮我整理 B 站收藏夹。下面是我收藏的全部视频，每行是「标题 | UP 主 | 时长」。

请先归纳这批内容涉及哪些主题，再把这些视频分成若干组，每组会变成一个收藏夹。只输出 JSON。

规则：
1. 一个游戏、一部作品只能出现在一个分组里。
   反例（不要这样）：A游戏攻略 / A游戏外观 / A游戏赛事 —— 同一个游戏被拆成三个。
   正例（要这样）：A游戏 —— 副本、外观、职业、同人全都放这里。
2. 分组的依据是「讲什么」，不是「怎么讲」：
   攻略、实况、二创、赛事、解说、杂谈，都是讲法，不是主题。
3. **组名里不许出现具体的人名、主播名、UP 主名，也不许用他们的团体名。**
   反例（不要这样）：阿强解说 / 某某后援团 / 某某的日常。
   主播相关的内容不按形态拆：直播录像、切片、鬼畜二创、赛事解说属于同一个主题，
   合成一组，用内容形态命名（例如「主播整活」「直播二创」）。
4. {GROUPS_RULE}**条数不决定建不建组**：一个主题只要在内容上独立，
   哪怕只有两三条，也要单独成组（例如「健身训练」只有三条，也该有自己的一组），
   不要因为条数少就并进别的主题；反过来也不要把一个大主题硬拆成几组来凑数。
5. 输出前自查一遍：把所有组名排在一起看，
   如果两个组名里含有同一个游戏名或同一部作品名，就把它们合并。
6. 不要兜底分类：「其他」「综合」「杂项」「其他游戏攻略」都算兜底。
7. 每个分组给：name（4-8 字的名词）、definition（一行定义，说明什么样的内容属于它）、
   examples（5 个真实标题原文，从上面的列表里原样摘取，不要改写）

格式（不要解释、不要代码块标记）：
{"categories": [{"name": "游戏名", "definition": "……", "examples": ["…", "…", "…", "…", "…"]}]}
"""

# Enough for the 50-category ceiling plus their definitions. A thinking model
# can still burn this on reasoning, so `content` emptiness is checked below.
MAX_TOKENS = 6000

DESCRIBE_PROMPT = """下面每组是一个收藏夹分类，以及它实际收到的视频标题（样本）。

请为每组写一句 20 字以内的描述，说明这一组实际装的是什么内容。
只写内容特征：讲什么主题、什么类型。不写评价，不要出现「本组」「这些视频」「该分类」
这类词，也不要只是重复分类名本身。

只输出 JSON：
{"descriptions": {"分类名": "描述"}}
"""


REVIEW_PROMPT = """下面是每个收藏夹分类（名字 + 定义），以及一些「位置不确定」的视频。

这些视频被算法分进了某一类，但相似度明显低于这一类里的其他内容，可能本来就不属于这里。

请逐条判断：它应该归到哪个分类，还是放进「存疑」。
只看标题讲什么；不要因为分数低就丢进存疑，也不要因为它已经被分到某类就默认留着。

规则：
1. target 必须是下面列出的分类名之一，或者「存疑」。
2. 标题里能看出明确主题、且下面有对应分类的，归过去。
3. 看不出主题，或者主题在分类里根本没有的，放「存疑」。

只输出 JSON：
{"decisions": [{"id": "视频id", "target": "分类名或存疑"}]}
"""


SHAPE_PROMPT = """下面是我准备建的一批 B 站收藏夹分类（名字 + 定义），现在有 {NOW} 个，目标是 {WANT} 个左右。

请检查一遍，输出需要合并的组。要做两件事：

1. **同一个来源被拆开的，合并**：同一个游戏、同一部作品、同一个 UP 主的不同形态
   （攻略 / 实况 / 二创 / 赛事），算同一组。不同主题不要合并。
2. **数量超了就继续合并**：合并之后如果还超过 {WANT} 个，把最该合并的继续合掉，
   直到不超过 {WANT} 个。宁可合得粗一点，也不要为了凑数把两个不相干的主题硬拼在一起。

合并后的要求：
- 名字要短（4-8 字），能涵盖组内所有内容，不许出现人名、主播名、UP 主名。
- **必须写一条新的 definition，覆盖组内所有内容**，不能只描述其中一个成员。
- 不需要合并的分类不要出现在输出里。
- 每个合并组至少 2 个成员。

只输出 JSON：
{"merge": [{"name": "新名字", "definition": "……", "members": ["旧名1", "旧名2"]}]}
如果没有需要合并的，输出 {"merge": []}
"""


class TaxonomyError(RuntimeError):
    pass


def build_lines(
    conn: sqlite3.Connection, limit: int = 2000, platform: str = "bilibili"
) -> list[str]:
    platform = platforms.normalise(platform)
    container_kind, item_kind = platforms.shape(platform)
    rows = conn.execute(
        """
        SELECT title, meta_json FROM item
         WHERE platform = ? AND kind = ?
         ORDER BY last_seen DESC LIMIT ?
        """,
        (platform, item_kind, limit),
    ).fetchall()

    lines = []
    for row in rows:
        title = (row["title"] or "").strip()
        if not title:
            continue
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except ValueError:
            meta = {}
        lines.append(itemtext.describe(title, meta))
    return lines


def _groups_rule(count: int) -> str:
    """One sentence, worded for the number. "1 组左右（可差一两个）" is nonsense."""
    if count <= 2:
        return f"只分 {count} 组，不再往下细分。"
    return f"组数控制在 {count} 个左右（可以差一两个）。"


def _group_count(count: object) -> int:
    try:
        value = int(count)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_GROUPS
    return max(MIN_GROUPS, min(MAX_GROUPS, value))


def suggest(
    conn: sqlite3.Connection,
    platform: str = "bilibili",
    count: object = None,
) -> tuple[list[dict], dict]:
    endpoint = models.endpoint("llm")
    if endpoint is None:
        raise TaxonomyError("还没有配置对话模型")

    platform = platforms.normalise(platform)
    groups = _group_count(count)
    system_prompt = PROMPT.replace("{GROUPS_RULE}", _groups_rule(groups))

    lines = build_lines(conn, platform=platform)
    if not lines:
        raise TaxonomyError("库里没有内容，先采集一次")

    # Mark the log so the returned figure covers this call only.
    mark = usage.watermark(conn)

    body: dict = {
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if endpoint["provider"] == "deepseek":
        # Thinking is on by default and eats the token budget before any answer
        # is written. Grouping titles does not need it.
        body["thinking"] = {"type": "disabled"}

    payload = _post(endpoint, body)
    usage.record(
        conn, "suggest", endpoint["provider"], endpoint["model"], payload.get("usage")
    )

    try:
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise TaxonomyError("模型返回的格式不对") from exc

    items = _parse(content, finish)
    # One pass folds duplicates and fits the number the person asked for.
    items = _shape(endpoint, items, groups)
    return items, usage.since(conn, mark)


def _post(endpoint: dict, body: dict) -> dict:
    """Post once; if the server rejects the optional fields, retry without them.

    `response_format` and `thinking` are not part of every OpenAI-compatible
    implementation, and a 400 is the normal way to find out.
    """
    url = f"{endpoint['base_url'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {endpoint['api_key']}"}

    attempts = [body, {k: v for k, v in body.items() if k not in ("response_format", "thinking")}]
    last_error = ""
    for attempt_no, payload_body in enumerate(attempts):
        try:
            response = httpx.post(url, json=payload_body, headers=headers, timeout=180.0)
        except httpx.HTTPError as exc:
            raise TaxonomyError(f"请求失败：{type(exc).__name__}") from exc

        if response.status_code == 200:
            return response.json()
        last_error = f"模型返回 {response.status_code}：{_error(response)}"
        # Only a 400 justifies dropping optional fields; anything else is final.
        if response.status_code != 400 or attempt_no == len(attempts) - 1:
            break

    raise TaxonomyError(last_error)


def load(conn: sqlite3.Connection) -> list[dict] | None:
    row = conn.execute("SELECT data_json FROM taxonomy WHERE id = 'current'").fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["data_json"])
    except ValueError:
        return None


def save(conn: sqlite3.Connection, items: list[dict]) -> None:
    cleaned = []
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        cleaned.append(
            {
                "name": name,
                "group": str(item.get("group") or "").strip(),
                "definition": str(item.get("definition") or "").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "examples": [str(x) for x in (item.get("examples") or [])][:5],
            }
        )
    if not cleaned:
        raise TaxonomyError("没有有效的分类")

    conn.execute(
        """
        INSERT INTO taxonomy (id, data_json, updated_at) VALUES ('current', ?, ?)
        ON CONFLICT(id) DO UPDATE SET data_json = excluded.data_json,
                                      updated_at = excluded.updated_at
        """,
        (json.dumps(cleaned, ensure_ascii=False), int(time.time())),
    )
    conn.commit()


def review(payload: dict) -> dict[str, str]:
    """Ask the model where the uncertain items belong.

    Cosine cannot tell "this fits, but the whole category scores low" from
    "this fits nothing here". A person can, from the title alone, and so can
    the model. Returns item id -> category name, or 存疑.
    """
    endpoint = models.endpoint("llm")
    if endpoint is None:
        return {}

    body = {
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": 4000,
        "temperature": 0,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if endpoint["provider"] == "deepseek":
        body["thinking"] = {"type": "disabled"}

    try:
        payload_out = _post(endpoint, body)
        content = payload_out["choices"][0]["message"].get("content") or ""
        data = _try_json(_strip_fence(content)) or {}
    except Exception:  # noqa: BLE001
        return {}

    out: dict[str, str] = {}
    for decision in data.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        item_id = str(decision.get("id") or "").strip()
        target = str(decision.get("target") or "").strip()
        if item_id and target:
            out[item_id] = target
    return out


def describe(samples: dict[str, list[str]]) -> dict[str, str]:
    """One description per category, written from what actually landed in it.

    The proposal's definition is written before anything is assigned, so it
    describes the intent, not the contents. This runs after assignment and
    looks only at titles, which is the difference between "摄影与后期" and a
    folder that turned out to be 90% 相机设置.
    """
    if not samples:
        return {}
    endpoint = models.endpoint("llm")
    if endpoint is None:
        return {}

    body = {
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": DESCRIBE_PROMPT},
            {"role": "user", "content": json.dumps(samples, ensure_ascii=False)},
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if endpoint["provider"] == "deepseek":
        body["thinking"] = {"type": "disabled"}

    try:
        payload = _post(endpoint, body)
        content = payload["choices"][0]["message"].get("content") or ""
        data = _try_json(_strip_fence(content)) or {}
    except Exception:  # noqa: BLE001
        # A missing description is cosmetic; the classification still stands.
        return {}

    out: dict[str, str] = {}
    for name, text in (data.get("descriptions") or {}).items():
        if not isinstance(name, str) or not isinstance(text, str):
            continue
        clean = " ".join(text.split())[:60]
        if clean:
            out[name] = clean
    return out


def _parse(content: str, finish_reason: str = "") -> list[dict]:
    text = (content or "").strip()
    if not text:
        if finish_reason == "length":
            raise TaxonomyError(
                "模型的输出被截断了，没写出任何分类。"
                "多半是思考模式占满了 token；换一个非思考模型，或把 max_tokens 调大。"
            )
        raise TaxonomyError("模型返回了空内容。检查一下模型名是否可用，或换个模型试试。")

    entries = _extract_entries(text)
    if entries is None:
        raise TaxonomyError(f"模型没有返回 JSON 数组。它说的是：{_shorten(text)}")

    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        group = str(entry.get("name") or "").strip()
        if not group:
            continue

        children = entry.get("children")
        if isinstance(children, list) and children:
            for child in children:
                item = _leaf(child, group)
                if item:
                    result.append(item)
            continue

        # The domain itself is the category: nothing under it to split.
        item = _leaf(entry, "")
        if item:
            result.append(item)
    if not result:
        raise TaxonomyError(f"模型给的东西里没有分类名。原样：{_shorten(text)}")
    return result


def _extract_entries(text: str) -> list | None:
    """Find the category list in whatever shape the model chose.

    Tried in order: the whole string as JSON, the same after removing a
    markdown fence, then any bracketed slice inside the text.
    """
    for candidate in (text, _strip_fence(text)):
        parsed = _try_json(candidate)
        if isinstance(parsed, dict):
            for field in ("groups", "categories", "items", "data", "list", "result"):
                value = parsed.get(field)
                if isinstance(value, list):
                    return value
        if isinstance(parsed, list):
            return parsed

    start, end = text.find("["), text.rfind("]")
    if 0 <= start < end:
        parsed = _try_json(text[start : end + 1])
        if isinstance(parsed, list):
            return parsed
    return None


def _shape(endpoint: dict, items: list[dict], count: int) -> list[dict]:
    """Merge duplicates and fit the count, in one call where possible.

    The two jobs were separate prompts, but they are the same operation on the
    same list: folding groups that belong together. Splitting them could not
    see the other's result, and cost a second call whenever the first left too
    many groups.

    The first round always runs, because there is always de-duplication to do.
    A second round runs only when the count is still over AND the first round
    actually merged something - asking again after "nothing to merge" only
    spends tokens.
    """
    for attempt in range(2):
        if attempt and len(items) <= count:
            break
        listing = [
            {"name": item["name"], "definition": item.get("definition") or ""}
            for item in items
        ]
        body = {
            "model": endpoint["model"],
            "messages": [
                {
                    "role": "system",
                    "content": SHAPE_PROMPT.replace("{NOW}", str(len(items))).replace(
                        "{WANT}", str(count)
                    ),
                },
                {"role": "user", "content": json.dumps(listing, ensure_ascii=False)},
            ],
            "max_tokens": 1500,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if endpoint["provider"] == "deepseek":
            body["thinking"] = {"type": "disabled"}

        try:
            payload = _post(endpoint, body)
            content = payload["choices"][0]["message"].get("content") or ""
            data = _try_json(_strip_fence(content)) or {}
        except Exception:  # noqa: BLE001
            # A failed pass must never cost the first result.
            return items

        merged = _apply_merges(items, data.get("merge") or [])
        if len(merged) >= len(items):
            return items
        items = merged
    return items


def _apply_merges(items: list[dict], groups: object) -> list[dict]:
    """Fold the model's merge groups back into one entry each."""
    by_name = {item["name"]: item for item in items}
    consumed: set[str] = set()
    merged: list[dict] = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        members = [
            name for name in (group.get("members") or [])
            if name in by_name and name not in consumed
        ]
        if len(members) < 2:
            continue
        host = dict(by_name[members[0]])
        new_name = str(group.get("name") or "").strip()
        if new_name:
            host["name"] = new_name
        examples: list[str] = []
        for name in members:
            examples.extend(by_name[name].get("examples") or [])
            consumed.add(name)
        host["examples"] = examples[:5]
        # The merged name and the merged definition must describe the same set.
        # Taking the longest member's definition is how a group called 绘画教程
        # ended up defined as 摄影教学: the name was new, the definition was one
        # member's. Prefer what the model wrote for the merged group, and fall
        # back to the longest only when it wrote nothing.
        supplied = str(group.get("definition") or "").strip()
        if supplied:
            host["definition"] = supplied
        else:
            longest = max(
                (by_name[name].get("definition") or "" for name in members), key=len
            )
            if longest:
                host["definition"] = longest
        merged.append(host)

    kept = [item for item in items if item["name"] not in consumed]
    return kept + merged


def _leaf(entry: object, group: str) -> dict | None:
    if not isinstance(entry, dict):
        return None
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "group": group,
        "definition": str(entry.get("definition") or "").strip(),
        "examples": [str(x) for x in (entry.get("examples") or [])][:5],
    }


def display_name(category: dict) -> str:
    """What the folder is actually called: 游戏 · Dota2."""
    group = str(category.get("group") or "").strip()
    name = str(category.get("name") or "").strip()
    if group and group != name:
        return f"{group} · {name}"
    return name


def _strip_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)


def _try_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _shorten(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def _error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:120]
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:120]
    return str(error or "")[:120]
