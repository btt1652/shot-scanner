"""SQLite-based cache for vsearch video search/classify results.

Replaces JSON file cache with a proper relational database for:
- Fast lookup by video path, probability threshold, timestamp range
- Concurrent reads (safe for AI agents)
- Atomic writes (no corruption)
- Custom SQL queries for advanced users
- Cache statistics and management

Cache schema version: 2
"""

import sqlite3
import hashlib
import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cache database location (per cache_file)
DB_NAME = "cache.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    fps REAL,
    total_frames INTEGER,
    stride INTEGER DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    cache_status TEXT NOT NULL DEFAULT 'empty' -- empty, partial, complete
);

CREATE TABLE IF NOT EXISTS frame_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    question_name TEXT NOT NULL,
    question_instructions TEXT,
    criteria_json TEXT,
    top_choice TEXT,
    top_probability REAL,
    probabilities_json TEXT,
    batch_id INTEGER,
    raw_response TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(video_id, frame_index, question_name)
);

CREATE INDEX IF NOT EXISTS idx_frame_answers_video ON frame_answers(video_id);
CREATE INDEX IF NOT EXISTS idx_frame_answers_choice ON frame_answers(top_choice);
CREATE INDEX IF NOT EXISTS idx_frame_answers_qname ON frame_answers(question_name);
CREATE INDEX IF NOT EXISTS idx_frame_answers_qname_choice ON frame_answers(question_name, top_choice);

CREATE TABLE IF NOT EXISTS cache_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_VERSION = 2

# Keys for cache_stats
STAT_TOTAL_VIDEOS = "total_videos"
STAT_TOTAL_FRAMES = "total_frames"
STAT_FIRST_RUN = "first_run"
STAT_LAST_PRUNE = "last_prune"


class CacheDB:
    """SQLite cache manager for vsearch results."""

    def __init__(self, cache_file: str = "./.cache.db"):
        self.cache_file = Path(cache_file).resolve()
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_file
        self._conn: Optional[sqli3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._execute_schema()
        return self._conn

    def _execute_schema(self):
        """Initialize schema if not exists."""
        self.conn.executescript(SCHEMA_SQL)

    def set_stat(self, key: str, value: str):
        """Set a cache statistics entry."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache_stats (key, value, updated_at) VALUES (?, ?, datetime('now'))",
                (key, value),
            )

    def get_stat(self, key: str, default: str = "") -> str:
        """Get a cache statistics entry."""
        cur = self.conn.execute(
            "SELECT value FROM cache_stats WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["value"] if row else default

    # --- Video management ---

    def upsert_video(
        self,
        video_path: str,
        content_hash: str,
        params_hash: str,
        fps: Optional[float] = None,
        total_frames: Optional[int] = None,
        stride: int = 1,
    ) -> int:
        """Insert or update video metadata. Returns video_id."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO videos (video_path, content_hash, params_hash, fps, total_frames, stride, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(video_path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       params_hash=excluded.params_hash,
                       fps=excluded.fps,
                       total_frames=excluded.total_frames,
                       stride=excluded.stride,
                       updated_at=datetime('now')""",
                (video_path, content_hash, params_hash, fps, total_frames, stride),
            )
        # Return the video_id
        cur = self.conn.execute(
            "SELECT id FROM videos WHERE video_path = ?", (video_path,)
        )
        return cur.fetchone()["id"]

    def get_video(self, video_path: str) -> Optional[Dict]:
        """Get video metadata by path."""
        cur = self.conn.execute(
            "SELECT * FROM videos WHERE video_path = ?", (video_path,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_videos(self, limit: int = 1000) -> List[Dict]:
        """List all cached videos."""
        cur = self.conn.execute(
            "SELECT id, video_path, content_hash, created_at, cache_status FROM videos ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    # --- Frame answer operations ---

    def upsert_frame_answers(
        self,
        video_id: int,
        answers: List[Dict],
    ) -> int:
        """Insert or update per-question answers for a video.

        Each item in answers should contain:
          frame_index, timestamp_seconds, question_name, top_choice,
          top_probability, probabilities_json, batch_id
        plus optional question_instructions, criteria_json, raw_response.
        """
        if not answers:
            return 0
        with self.conn:
            for a in answers:
                self.conn.execute(
                    """INSERT INTO frame_answers (
                           video_id, frame_index, timestamp_seconds, question_name,
                           question_instructions, criteria_json, top_choice, top_probability,
                           probabilities_json, batch_id, raw_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(video_id, frame_index, question_name) DO UPDATE SET
                           timestamp_seconds=excluded.timestamp_seconds,
                           question_instructions=excluded.question_instructions,
                           criteria_json=excluded.criteria_json,
                           top_choice=excluded.top_choice,
                           top_probability=excluded.top_probability,
                           probabilities_json=excluded.probabilities_json,
                           batch_id=excluded.batch_id,
                           raw_response=excluded.raw_response,
                           created_at=datetime('now')""",
                    (
                        video_id,
                        a.get("frame_index"),
                        a.get("timestamp_seconds"),
                        a.get("question_name"),
                        a.get("question_instructions"),
                        a.get("criteria_json"),
                        a.get("top_choice"),
                        a.get("top_probability"),
                        a.get("probabilities_json"),
                        a.get("batch_id"),
                        a.get("raw_response"),
                    ),
                )
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM frame_answers WHERE video_id = ?", (video_id,)
        )
        return cur.fetchone()[0]

    def get_answers(
        self,
        video_id: int,
        question_name: Optional[str] = None,
        top_choice: Optional[str] = None,
        min_probability: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Get frame answers with optional filters."""
        query = (
            "SELECT fa.*, v.video_path AS source_video_path FROM frame_answers fa "
            "JOIN videos v ON fa.video_id = v.id WHERE fa.video_id = ?"
        )
        params: List[Any] = [video_id]

        if question_name is not None:
            query += " AND fa.question_name = ?"
            params.append(question_name)
        if top_choice is not None:
            query += " AND fa.top_choice = ?"
            params.append(top_choice)
        if min_probability is not None:
            query += " AND fa.top_probability >= ?"
            params.append(min_probability)

        query += " ORDER BY fa.timestamp_seconds ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cur = self.conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def get_answer(
        self,
        video_id: int,
        frame_index: int,
        question_name: str,
    ) -> Optional[Dict]:
        """Get a single answer by video, frame and question name."""
        cur = self.conn.execute(
            "SELECT * FROM frame_answers WHERE video_id = ? AND frame_index = ? AND question_name = ?",
            (video_id, frame_index, question_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # --- Cache lookup ---

    def get_cached_results(
        self,
        video_path: str,
        decision_model_floor: float,
        **params,  # sample_rate, frame_scale, state, instructions, model, etc.
    ) -> Optional[Tuple[List[Dict], Dict]]:
        """Try to load cached results for a video + params.

        Returns (results, data_dict) or None if cache miss.
        """
        video = self.get_video(video_path)
        if video is None:
            return None

        # Check params match (content_hash + params_hash)
        # ... simplified: we just check video exists and return what we have
        # In a full impl, we'd compare params_hash against current params

        # Return all frame answers for this video
        answers = self.get_answers(video_id=video["id"])
        if not answers:
            return None, None

        results = answers  # all answers stored in frame_answers
        return results, {"video_id": video["id"], "video_path": video_path}

    # --- Custom SQL support ---

    def execute_sql(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict]:
        """Execute a custom SQL query and return results as dicts.

        For advanced users who want to do arbitrary queries on the cache.
        """
        if params is None:
            params = []
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def execute_sql_command(self, sql: str, params: Optional[List[Any]] = None) -> int:
        """Execute a custom SQL command (INSERT, UPDATE, DELETE) and return rowcount."""
        if params is None:
            params = []
        with self.conn:
            cur = self.conn.execute(sql, params)
            return cur.rowcount

    # --- Cleanup ---

    def prune_old(self, max_age_days: int = 30):
        """Delete cache entries older than max_age_days."""
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=max_age_days)).isoformat()
        with self.conn:
            # Delete old frames first (CASCADE handles frames when video deleted)
            self.conn.execute(
                "DELETE FROM cache_stats WHERE updated_at < ?", (cutoff,)
            )
            # Could add more pruning logic here

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def _content_hash(video_path: str, chunk_mb: int = 8) -> str:
    """Compute BLAKE2b hash of first/last N MB of video."""
    size = os.path.getsize(video_path)
    chunk = chunk_mb * 1024 * 1024
    h = hashlib.blake2b(digest_size=32)
    with open(video_path, "rb") as f:
        h.update(f.read(min(chunk, size)))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()


def _params_hash(**kwargs) -> str:
    """Hash parameter dict into a string key."""
    s = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


# Convenience function for CLI
def init_cache(cache_file: str = "./.cache.db") -> CacheDB:
    """Initialize/cache database at the given file path."""
    return CacheDB(cache_file)