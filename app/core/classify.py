"""Assign every item to a category by embedding similarity.

The model never labels individual items; it only names categories. Assignment
is arithmetic: embed each category description, embed each item, take the
nearest one, and keep the similarity as confidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import statistics
import time

import httpx

from . import itemtext, models, platforms, taxonomy, usage

# An item scoring below this fraction of its own category's median is where the
# taxonomy probably does not cover it. It is a candidate, not a verdict: CS2
# 道具 sits at 32% and is right, 臀桥 sits at 34% and is wrong.
OUTLIER_RATIO = 0.75
# One model call reviews this many at most, worst first.
REVIEW_MAX = 60
# A category is a theme, not a size. The only floor is one item: a folder
# holding a single video is noise in the list, so that one video goes to 存疑.
# Everything above it stays, however small - a three-item 健身训练 is still the
# only correct home for those three videos.
MIN_ITEMS = 2

# The container for songs that did not earn a category of their own.
PARKING_NAME = "存疑"

# Bilibili returns this exact title for a video that is gone, and every such
# row is identical text. Embedding them is meaningless: they are one vector,
# so they all land in whichever category happens to be nearest and turn it
# into a dump. They never go to a category.
DEAD_TITLES = {"已失效视频", "已删除视频"}

BATCH = 32


log = logging.getLogger(__name__)


class ClassifyError(RuntimeError):
    pass


def run(conn: sqlite3.Connection, platform: str = "bilibili") -> dict:
    platform = platforms.normalise(platform)
    endpoint = models.endpoint("embedding")
    if endpoint is None:
        raise ClassifyError("还没有配置向量模型")

    categories = taxonomy.load(conn)
    if not categories:
        raise ClassifyError("先建好分类体系")

    items = _load_items(conn, platform)
    if not items:
        raise ClassifyError("库里没有内容，先采集一次")

    mark = usage.watermark(conn)

    cat_vectors = _embed(conn, endpoint, [_category_text(c) for c in categories])
    item_vectors = _embed(conn, endpoint, [item["text"] for item in items])

    assignments = []
    tops: list[float] = []
    for item, vector in zip(items, item_vectors):
        scores = [_cosine(vector, cat) for cat in cat_vectors]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        best = order[0]
        top = scores[best]
        runner = scores[order[1]] if len(order) > 1 else 0.0
        tops.append(top)

        second = categories[order[1]]["name"] if len(order) > 1 else ""
        # A deleted video has no text worth matching on; its score is noise.
        # Putting it in a category is what filled 板绘厚涂教程 with 11 of them.
        dead = item["title"].strip() in DEAD_TITLES
        assignments.append(
            {
                "item_id": item["item_id"],
                "target": PARKING_NAME if dead else taxonomy.display_name(categories[best]),
                "confidence": round(top, 4),
                # The runner-up is what makes a score judgeable: 38% means
                # nothing until you see the other option was 37%.
                "reason": f"{second}|{runner:.4f}",
            }
        )

    # Every item gets its nearest category; what happens next is about size,
    # not confidence. A category holding three songs is not a playlist anyone
    # will open, and folding those three into a bigger category would file them
    # under something they are not. They go to 存疑 and stay visible there.
    names_before = [taxonomy.display_name(c) for c in categories]
    assignments, thin = _park_small(assignments, MIN_ITEMS)
    if thin:
        # A category that could not hold MIN_ITEMS songs is not a suggestion
        # worth showing. Keeping it would put a "0 条" row in the list and
        # invite the person to wonder what it was for.
        categories = [
            c for c in categories if taxonomy.display_name(c) not in thin
        ]

    # Anything the taxonomy does not really cover has to land somewhere; ask
    # the model whether it belongs at all, instead of letting the nearest
    # category absorb it and then blaming the category's name.
    kept = {taxonomy.display_name(c) for c in categories}
    vec_by_name = {
        name: vector
        for name, vector in zip(names_before, cat_vectors)
        if name in kept
    }
    assignments, review = _review(
        categories, items, assignments, item_vectors, vec_by_name
    )

    # Descriptions are written last, from what each folder actually received.
    # The suggestion's definition was a guess made before any assignment.
    categories = _describe(categories, items, assignments)
    taxonomy.save(conn, categories)

    _save_plan(conn, categories, assignments, platform)
    # Read back instead of summarising twice: one shape for the live response
    # and for /api/plan, so the page cannot see two different truths.
    result = plan(conn, platform) or {}
    result["usage"] = usage.since(conn, mark)
    result["scores"] = _spread(tops)
    result["review"] = review
    return result



def retarget_orphans(conn: sqlite3.Connection, keep: list[str]) -> int:
    """Drop plan rows whose category no longer exists.

    The items stay exactly where they are — a deleted category must not delete
    its songs, and it must not leave them pointing at a name that no longer
    exists either. There is no parking folder to dump them into, so they leave
    the plan instead. A plan only contains moves the person can still see a
    destination for.
    """
    allowed = {name for name in keep if name}
    rows = conn.execute(
        "SELECT DISTINCT target FROM plan_item WHERE plan_id = 'current'"
    ).fetchall()
    orphans = [
        row["target"] for row in rows
        if row["target"] and row["target"] not in allowed
    ]
    if not orphans:
        return 0

    dropped = 0
    for name in orphans:
        cursor = conn.execute(
            "DELETE FROM plan_item WHERE plan_id = 'current' AND target = ?",
            (name,),
        )
        dropped += cursor.rowcount
    conn.commit()
    return dropped


def _review(
    categories: list[dict],
    items: list[dict],
    assignments: list[dict],
    item_vectors: list[list[float]],
    vec_by_name: dict[str, list[float]],
) -> tuple[list[dict], dict]:
    """Send the items that do not fit their category to the model for a verdict.

    The candidate list is built from the data, not from a fixed score: an item
    is a candidate when it scores well below the median of the very category it
    landed in. That keeps 沙二双烟 out of it (its category runs low) and puts
    臀桥 in it (its category runs high).
    """
    by_target: dict[str, list[float]] = {}
    for a in assignments:
        if a["target"] != PARKING_NAME:
            by_target.setdefault(a["target"], []).append(a["confidence"] or 0.0)

    medians = {name: statistics.median(values) for name, values in by_target.items()}
    candidates = [
        a
        for a in assignments
        if a["target"] != PARKING_NAME
        and medians.get(a["target"], 0) > 0
        and (a["confidence"] or 0) < medians[a["target"]] * OUTLIER_RATIO
    ]
    if not candidates:
        return assignments, {"checked": 0, "parked": 0, "moved": 0}
    candidates.sort(key=lambda a: a["confidence"] or 0)
    candidates = candidates[:REVIEW_MAX]

    titles = {item["item_id"]: item["title"] for item in items}
    payload = {
        "categories": [
            {
                "name": taxonomy.display_name(c),
                "definition": c.get("definition") or "",
            }
            for c in categories
        ],
        "items": [
            {
                "id": a["item_id"],
                "title": titles.get(a["item_id"], ""),
                "current": a["target"],
            }
            for a in candidates
        ],
    }
    verdicts = taxonomy.review(payload)
    if not verdicts:
        return assignments, {"checked": len(candidates), "parked": 0, "moved": 0}

    allowed = {taxonomy.display_name(c) for c in categories} | {PARKING_NAME}
    positions = {a["item_id"]: i for i, a in enumerate(assignments)}
    parked = moved = 0
    for a in candidates:
        target = verdicts.get(a["item_id"])
        if not target or target not in allowed:
            continue
        if target == a["target"]:
            continue
        if target == PARKING_NAME:
            parked += 1
        else:
            moved += 1
        a["target"] = target

        # The score was measured against the old category; measure it against
        # the one it is going to, and keep the runner-up honest too.
        vector = item_vectors[positions[a["item_id"]]]
        scores = sorted(
            ((_cosine(vector, cat), name) for name, cat in vec_by_name.items()),
            reverse=True,
        )
        hit = next((s for s, n in scores if n == target), None)
        if hit is not None:
            a["confidence"] = round(hit, 4)
            rest = [(s, n) for s, n in scores if n != target]
            if rest:
                a["reason"] = f"{rest[0][1]}|{rest[0][0]:.4f}"

    return assignments, {
        "checked": len(candidates),
        "parked": parked,
        "moved": moved,
    }


def _describe(
    categories: list[dict], items: list[dict], assignments: list[dict]
) -> list[dict]:
    """Attach a summary to each category, written from its own contents.

    Eight highest-confidence titles per category is enough for the model to
    say what the folder is about, and short enough to keep this to one call.
    """
    by_id = {item["item_id"]: item for item in items}
    samples: dict[str, list[str]] = {}
    for a in sorted(assignments, key=lambda x: -(x.get("confidence") or 0)):
        if a["target"] == PARKING_NAME:
            continue
        titles = samples.setdefault(a["target"], [])
        if len(titles) >= 8:
            continue
        title = (by_id.get(a["item_id"]) or {}).get("title") or ""
        if title.strip():
            titles.append(title.strip())

    written = taxonomy.describe({k: v for k, v in samples.items() if v})
    out = []
    for category in categories:
        summary = written.get(taxonomy.display_name(category))
        out.append(dict(category, summary=summary) if summary else category)
    return out


def _park_small(assignments: list[dict], minimum: int) -> tuple[list[dict], set[str]]:
    """Send the songs of undersized categories to 存疑.

    Counted by target name, which is what a folder is called. Nothing is
    merged: merging says "these songs belong with those", and the only thing
    known about them is that there are few.
    """
    counts: dict[str, int] = {}
    for assignment in assignments:
        target = assignment["target"]
        counts[target] = counts.get(target, 0) + 1

    thin = {
        name for name, count in counts.items()
        if count < minimum and name != PARKING_NAME
    }
    if not thin:
        return assignments, set()

    parked = 0
    for assignment in assignments:
        if assignment["target"] in thin:
            assignment["target"] = PARKING_NAME
            parked += 1
    log.info("parked %s songs from %s undersized categories", parked, len(thin))
    return assignments, thin


def _spread(values: list[float]) -> dict | None:
    """What the scores actually looked like, so the cut-offs can be judged."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return {
        "min": round(ordered[0], 4),
        "median": round(median, 4),
        "max": round(ordered[-1], 4),
    }


def plan(conn: sqlite3.Connection, platform: str = "bilibili") -> dict | None:
    """Latest assignment, grouped by target, including empty categories.

    A plan belongs to one platform. Left over from the other one, it would
    describe items that are not in the library being looked at.
    """
    platform = platforms.normalise(platform)
    owner = conn.execute(
        "SELECT platform FROM plan WHERE plan_id = 'current'"
    ).fetchone()
    if owner and owner["platform"] != platform:
        return None

    rows = conn.execute(
        """
        SELECT target,
               COUNT(*) AS n,
               SUM(CASE WHEN action = 'move' THEN 1 ELSE 0 END) AS pending,
               AVG(confidence) AS confidence
          FROM plan_item
         WHERE plan_id = 'current'
         GROUP BY target
        """
    ).fetchall()
    if not rows:
        return None

    counts = {row["target"]: row["n"] for row in rows}
    pending = {row["target"]: row["pending"] for row in rows}
    mean = {row["target"]: row["confidence"] for row in rows}
    parked = counts.pop(PARKING_NAME, 0)
    parked_pending = pending.pop(PARKING_NAME, 0)
    parked_mean = mean.pop(PARKING_NAME, 0.0)

    # Use the taxonomy as it stands now, not the snapshot taken when the plan
    # was built: a category deleted since then must not come back as a zero row.
    for category in taxonomy.load(conn) or []:
        counts.setdefault(taxonomy.display_name(category), 0)
        pending.setdefault(taxonomy.display_name(category), 0)

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    total = sum(counts.values()) + parked

    return {
        "total": total,
        # How much of this plan has already been written back. The page needs
        # it to say 「继续执行」and to give the right counts.
        "moved": total - (sum(pending.values()) + parked_pending),
        "pending": parked,
        # A real folder like the others, but not a category: never editable.
        "parking": (
            {
                "name": PARKING_NAME,
                "count": parked,
                "remaining": parked_pending,
                "confidence": round(parked_mean, 3),
            }
            if parked
            else None
        ),
        "targets": [
            {
                "name": name,
                "count": count,
                "remaining": pending.get(name, 0),
                "confidence": round(mean.get(name, 0.0), 3),
            }
            for name, count in ordered
        ],
    }


def _load_items(conn: sqlite3.Connection, platform: str = "bilibili") -> list[dict]:
    platform = platforms.normalise(platform)
    container_kind, item_kind = platforms.shape(platform)
    rows = conn.execute(
        """
        SELECT i.item_id, i.title, i.meta_json
          FROM item i
         WHERE i.platform = ? AND i.kind = ?
           AND NOT EXISTS (
                SELECT 1
                  FROM membership m
                  JOIN container c ON c.platform = m.platform
                                  AND c.kind = m.container_kind
                                  AND c.container_id = m.container_id
                 WHERE m.platform = ? AND m.item_kind = ?
                   AND m.item_id = i.item_id
                   -- Only what the person ticked off. A folder that happens
                   -- to be called 存疑 is theirs unless they say otherwise:
                   -- matching on the name would swallow their items without
                   -- a word, and leave the checkbox saying it takes part.
                   AND c.keep_out = 1
           )
         ORDER BY i.last_seen DESC
        """,
        (platform, item_kind, platform, item_kind),
    ).fetchall()

    items = []
    for row in rows:
        title = (row["title"] or "").strip()
        if not title:
            continue
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except ValueError:
            meta = {}
        items.append({
            "item_id": row["item_id"],
            "title": title,
            "text": itemtext.describe(title, meta),
        })
    return items


def drop_container(
    conn: sqlite3.Connection, container_id: str, platform: str = "bilibili"
) -> int:
    """Take one folder's items out of the current plan.

    Excluding a folder after a classification has already run must not leave
    its items queued for a move: the plan would then contradict the exclusion.
    Returns how many rows were dropped.
    """
    platform = platforms.normalise(platform)
    container_kind, item_kind = platforms.shape(platform)
    rows = conn.execute(
        """
        SELECT m.item_id
          FROM membership m
         WHERE m.platform = ? AND m.container_kind = ? AND m.container_id = ?
        """,
        (platform, container_kind, str(container_id)),
    ).fetchall()
    dropped = 0
    for row in rows:
        cur = conn.execute(
            "DELETE FROM plan_item WHERE plan_id = 'current' AND item_id = ?",
            (row["item_id"],),
        )
        dropped += cur.rowcount
    conn.commit()
    return dropped


def _category_text(category: dict) -> str:
    parts = [taxonomy.display_name(category), category.get("definition", "")]
    examples = category.get("examples") or []
    if examples:
        parts.append("例如：" + "；".join(examples))
    return "。".join(part for part in parts if part)


def _embed(conn: sqlite3.Connection, endpoint: dict, texts: list[str]) -> list[list[float]]:
    url = f"{endpoint['base_url'].rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {endpoint['api_key']}"}
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        try:
            response = httpx.post(
                url,
                json={"model": endpoint["model"], "input": chunk},
                headers=headers,
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            raise ClassifyError(f"向量接口连不上：{type(exc).__name__}") from exc

        if response.status_code != 200:
            raise ClassifyError(f"向量接口返回 {response.status_code}：{_error(response)}")

        payload = response.json()
        usage.record(
            conn, "classify", endpoint["provider"], endpoint["model"], payload.get("usage")
        )
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(chunk):
            raise ClassifyError("向量接口返回的条数对不上")

        for entry in sorted(data, key=lambda item: item.get("index", 0)):
            vector = entry.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ClassifyError("向量接口没有返回 embedding")
            vectors.append(vector)

    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ClassifyError("向量维度不一致，换过模型吗？")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _save_plan(
    conn: sqlite3.Connection,
    categories: list[dict],
    assignments: list[dict],
    platform: str = "bilibili",
) -> None:
    payload = json.dumps(categories, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    conn.execute(
        """
        INSERT INTO plan (plan_id, platform, taxonomy_json, created_at, frozen_at, sha256)
        VALUES ('current', ?, ?, ?, NULL, ?)
        ON CONFLICT(plan_id) DO UPDATE SET
            taxonomy_json = excluded.taxonomy_json,
            created_at    = excluded.created_at,
            sha256        = excluded.sha256
        """,
        (platforms.normalise(platform), payload, int(time.time()), digest),
    )
    conn.execute("DELETE FROM plan_item WHERE plan_id = 'current'")
    conn.executemany(
        """
        INSERT INTO plan_item (plan_id, item_id, target, confidence, reason, action)
        VALUES ('current', ?, ?, ?, ?, 'move')
        """,
        [(a["item_id"], a["target"], a["confidence"], a["reason"]) for a in assignments],
    )
    conn.commit()


def _error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:120]
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:120]
    return str(error or "")[:120]


def items_of(
    conn: sqlite3.Connection,
    target: str,
    limit: int = 200,
    platform: str = "bilibili",
) -> dict:
    """The items a category is about to receive, best match first."""
    platform = platforms.normalise(platform)
    container_kind, item_kind = platforms.shape(platform)
    total = conn.execute(
        "SELECT COUNT(*) FROM plan_item WHERE plan_id = 'current' AND target = ?",
        (target,),
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT p.item_id, p.confidence, p.reason, i.title, i.meta_json
          FROM plan_item p
          LEFT JOIN item i ON i.platform = ?
                         AND i.kind = ?
                         AND i.item_id = p.item_id
         WHERE p.plan_id = 'current' AND p.target = ?
         ORDER BY p.confidence DESC
         LIMIT ?
        """,
        (platform, item_kind, target, limit),
    ).fetchall()

    items = []
    for row in rows:
        second, second_score = _runner(row["reason"])
        items.append(
            {
                "id": row["item_id"],
                "title": (row["title"] or "").strip() or "（无标题）",
                "upper": itemtext.upper_name(_meta(row["meta_json"])),
                "confidence": round(row["confidence"] or 0.0, 3),
                "runner": second,
                "runner_score": second_score,
            }
        )
    return {"total": total, "items": items}


def _runner(reason: str | None) -> tuple[str, float]:
    """The second-best category and its score, stored in `reason`."""
    text = reason or ""
    if "|" not in text:
        return "", 0.0
    name, _, raw = text.partition("|")
    try:
        return name, round(float(raw), 4)
    except ValueError:
        return name, 0.0


def _meta(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
