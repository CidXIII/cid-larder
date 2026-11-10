"""What the one supported platform calls its buckets and its contents.

Kept as a module rather than inlined constants because the queries read better
this way, and because a second platform would land here first.
"""

from __future__ import annotations

PLATFORMS = ("bilibili",)

CONTAINER_KIND = {"bilibili": "favorites"}
ITEM_KIND = {"bilibili": "video"}
LABEL = {"bilibili": "B 站"}


def shape(platform: str) -> tuple[str, str]:
    """(container kind, item kind) for a platform. Unknown platforms fail loudly."""
    if platform not in CONTAINER_KIND:
        raise ValueError(f"未知平台：{platform}")
    return CONTAINER_KIND[platform], ITEM_KIND[platform]


def normalise(platform: str | None) -> str:
    """Empty means bilibili: the page has only ever had one platform."""
    value = (platform or "").strip().lower()
    return value if value in CONTAINER_KIND else "bilibili"
