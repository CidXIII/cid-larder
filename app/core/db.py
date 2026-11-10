"""SQLite schema. One file, no ORM.

Three tables describe what was collected (container / item / membership); the
rest describe what the tool plans to do about it (plan / plan_item / run_log /
snapshot / backup).
"""

import sqlite3

# Everything lives in memory. Nothing collected is ever written to disk, so the
# next person to open the app cannot see what the last one had.
_MEMORY_URI = "file:cid-larder?mode=memory&cache=shared"
_keeper: sqlite3.Connection | None = None

SCHEMA_VERSION = 1

SCHEMA = """
-- A bucket: a favourites folder, a playlist.
CREATE TABLE IF NOT EXISTS container (
    platform     TEXT NOT NULL,
    kind         TEXT NOT NULL,          -- favorites | playlist
    container_id TEXT NOT NULL,
    name         TEXT,
    item_count   INTEGER,
    is_default   INTEGER NOT NULL DEFAULT 0,
    keep_out     INTEGER NOT NULL DEFAULT 0,   -- left out of the next run
    created      INTEGER NOT NULL DEFAULT 0,   -- made here, so safe to reuse
    seen_at      INTEGER,
    PRIMARY KEY (platform, kind, container_id)
);

-- A thing inside a bucket: a video, a song.
CREATE TABLE IF NOT EXISTS item (
    platform   TEXT NOT NULL,
    kind       TEXT NOT NULL,            -- video | song
    item_id    TEXT NOT NULL,
    title      TEXT,
    meta_json  TEXT,
    first_seen INTEGER NOT NULL,
    last_seen  INTEGER NOT NULL,
    state      TEXT NOT NULL DEFAULT 'fetched',
    PRIMARY KEY (platform, kind, item_id)
);

-- Which bucket a thing sits in. One item may sit in several buckets.
-- Container kind is stored too: ids are only unique within a kind, so
-- (container_id, item_id) alone is not unique.
CREATE TABLE IF NOT EXISTS membership (
    platform       TEXT NOT NULL,
    container_kind TEXT NOT NULL,        -- favorites | playlist
    container_id   TEXT NOT NULL,
    item_kind      TEXT NOT NULL,        -- video | song
    item_id        TEXT NOT NULL,
    position       INTEGER,
    added_at       INTEGER,
    PRIMARY KEY (platform, container_kind, container_id, item_kind, item_id)
);
CREATE INDEX IF NOT EXISTS idx_membership_item
    ON membership(platform, item_kind, item_id);

CREATE TABLE IF NOT EXISTS taxonomy (
    id         TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- One row per API call that can report token usage. NULL means the provider
-- did not tell us, which is different from "it was free".
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    stage             TEXT NOT NULL,          -- suggest | classify
    provider          TEXT,
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    cached_tokens     INTEGER,               -- prompt cache hits, billed apart
    ts                INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_stage ON usage_log(stage);

CREATE TABLE IF NOT EXISTS snapshot (
    run_id       TEXT NOT NULL,
    platform     TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    container_id TEXT,
    fav_time     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_snapshot_run ON snapshot(run_id);

CREATE TABLE IF NOT EXISTS plan (
    plan_id       TEXT PRIMARY KEY,
    platform      TEXT NOT NULL,
    taxonomy_json TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    frozen_at     INTEGER,
    sha256        TEXT
);

CREATE TABLE IF NOT EXISTS plan_item (
    plan_id    TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    target     TEXT,
    confidence REAL,
    reason     TEXT,
    action     TEXT,
    PRIMARY KEY (plan_id, item_id)
);

CREATE TABLE IF NOT EXISTS run_log (
    run_id    TEXT NOT NULL,
    plan_id   TEXT,
    batch_no  INTEGER,
    item_id   TEXT,
    action    TEXT,
    http_code INTEGER,
    verified  INTEGER,
    ts        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_log_run ON run_log(run_id);

CREATE TABLE IF NOT EXISTS backup (
    run_id       TEXT NOT NULL,
    platform     TEXT NOT NULL,
    container_id TEXT,
    items_json   TEXT NOT NULL,
    created_at   INTEGER NOT NULL
);
"""


def _keeper_connection() -> sqlite3.Connection:
    """One long-lived connection holds the shared in-memory database open.

    With `cache=shared` every other connection sees the same data, but the
    database disappears the moment the last one closes -- so this one is never
    closed while the process runs.
    """
    global _keeper
    if _keeper is None:
        _keeper = sqlite3.connect(_MEMORY_URI, uri=True, check_same_thread=False)
        _keeper.row_factory = sqlite3.Row
        _keeper.executescript(SCHEMA)
        _keeper.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        _keeper.commit()
    return _keeper


def connect() -> sqlite3.Connection:
    _keeper_connection()
    conn = sqlite3.connect(_MEMORY_URI, uri=True, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    """Create tables. Idempotent; safe to call on every start."""
    _keeper_connection()
    own = conn is None
    conn = conn or connect()
    try:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
    finally:
        if own:
            conn.close()
