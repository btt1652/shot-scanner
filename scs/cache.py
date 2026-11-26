"""SQLite-based cache for scs video search/classify results.

Replaces JSON file cache with a proper relational database for:
- Fast lookup by video path, probability threshold, timestamp range
- Concurrent reads (safe for AI agents)
- Atomic writes (no corruption)
- Custom SQL queries for advanced users
- Cache statistics and management
- Multiple analysis runs per video (via analysis_runs table)

Cache schema version: 3
"""

import sqlite3
import hashlib
import os
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cache database location (per cache)
DB_NAME = "cache.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    params_hash TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    questions_json TEXT NOT NULL DEFAULT '[]',
    cache_status TEXT NOT NULL DEFAULT 'empty', -- empty, partial, complete
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(video_id, params_hash)
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_video ON analysis_runs(video_id);

CREATE TABLE IF NOT EXISTS frame_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    question_name TEXT NOT NULL,
    question_instructions TEXT,
    criteria_json TEXT,
    top_choice TEXT,
    top_probability REAL,
    confidence REAL,  -- confidence from decision model response (e.g., model certainty)
    probabilities_json TEXT,
    raw_response TEXT,  -- raw decision model response JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, frame_index, question_name)
);

CREATE INDEX IF NOT EXISTS idx_frame_answers_run ON frame_answers(run_id);
CREATE INDEX IF NOT EXISTS idx_frame_answers_choice ON frame_answers(top_choice);
CREATE INDEX IF NOT EXISTS idx_frame_answers_qname ON frame_answers(question_name);
CREATE INDEX IF NOT EXISTS idx_frame_answers_qname_choice ON frame_answers(question_name, top_choice);

CREATE TABLE IF NOT EXISTS cache_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_VERSION = 6

# Keys for cache_stats
STAT_TOTAL_VIDEOS = "total_videos"
STAT_TOTAL_FRAMES = "total_frames"
STAT_FIRST_RUN = "first_run"
STAT_LAST_PRUNE = "last_prune"


class CacheDB:
    """SQLite cache manager for scs results."""

    def __init__(self, cache: str = "./.cache.db"):
        self.cache = Path(cache).resolve()
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._execute_schema()
        return self._conn

    def _execute_schema(self):
        """Ensure database schema matches SCHEMA_SQL by dropping all tables and recreating."""
        # Drop all user tables so SCHEMA_SQL creates them fresh from the current spec.
        # No backward compatibility — stale DBs always get wiped.
        self.conn.executescript("""
            DROP TABLE IF EXISTS frame_answers;
            DROP TABLE IF EXISTS analysis_runs;
            DROP TABLE IF EXISTS videos;
        """)
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
    ) -> int:
        """Insert or update video metadata. Returns video_id."""
        with self.conn:
            # Delete existing row if it exists
            self.conn.execute(
                "DELETE FROM videos WHERE video_path = ?", (video_path,)
            )
            self.conn.execute(
                """INSERT INTO videos (video_path, content_hash, updated_at)
                   VALUES (?, ?, datetime('now'))""",
                (video_path, content_hash),
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
        """List all cached videos with run info."""
        cur = self.conn.execute(
            """SELECT v.id, v.video_path, v.content_hash, v.created_at, v.updated_at,
                      COUNT(DISTINCT ar.id) AS run_count,
                      SUM(CASE WHEN ar.cache_status = 'complete' THEN 1 ELSE 0 END) AS complete_runs
               FROM videos v
               LEFT JOIN analysis_runs ar ON v.id = ar.video_id
               GROUP BY v.id
               ORDER BY v.updated_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Add latest run cache_status
        for r in rows:
            cur2 = self.conn.execute(
                "SELECT cache_status FROM analysis_runs WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
                (r["id"],)
            )
            rr = cur2.fetchone()
            r["cache_status"] = rr["cache_status"] if rr else "empty"
        return rows

    def list_runs(self, video_path: str) -> List[Dict]:
        """List all analysis runs for a specific video."""
        video = self.get_video(video_path)
        if not video:
            return []
        cur = self.conn.execute(
            """SELECT ar.id, ar.content_hash, ar.params_hash, ar.model, ar.cache_status,
                      ar.created_at, ar.updated_at,
                      COUNT(fa.id) AS answer_count
               FROM analysis_runs ar
               LEFT JOIN frame_answers fa ON ar.id = fa.run_id
               WHERE ar.video_id = ?
               GROUP BY ar.id
               ORDER BY ar.created_at DESC""",
            (video["id"],),
        )
        return [dict(r) for r in cur.fetchall()]

    def upsert_run(
        self,
        video_id: int,
        params_hash: str,
        model: str = '',
        questions_json: str = '[]',
    ) -> int:
        """Insert or update an analysis run. Returns run_id."""
        with self.conn:
            # Delete existing row if it exists
            self.conn.execute(
                "DELETE FROM analysis_runs WHERE video_id = ? AND params_hash = ?",
                (video_id, params_hash),
            )
            self.conn.execute(
                """INSERT INTO analysis_runs (video_id, params_hash, model, questions_json, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (video_id, params_hash, model, questions_json),
            )
        cur = self.conn.execute(
            "SELECT id FROM analysis_runs WHERE video_id = ? AND params_hash = ?",
            (video_id, params_hash),
        )
        return cur.fetchone()["id"]

    def get_run(self, video_id: int, params_hash: str) -> Optional[Dict]:
        """Get analysis run by video and params hash."""
        cur = self.conn.execute(
            "SELECT * FROM analysis_runs WHERE video_id = ? AND params_hash = ?",
            (video_id, params_hash),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_run_status(self, run_id: int, status: str):
        """Update analysis run cache_status."""
        with self.conn:
            self.conn.execute(
                "UPDATE analysis_runs SET cache_status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, run_id),
            )

    # --- Frame answer operations ---

    def upsert_frame_answers(
        self,
        run_id: int,
        answers: List[Dict],
    ) -> int:
        """Insert or update per-question answers for an analysis run.

        Each item in answers should contain:
          frame_index, timestamp_seconds, question_name, top_choice,
          top_probability, probabilities_json, confidence (optional)
        plus optional question_instructions, criteria_json, raw_response.
        """
        if not answers:
            return 0
        with self.conn:
            # Delete existing answers for this run to avoid conflicts
            self.conn.execute(
                "DELETE FROM frame_answers WHERE run_id = ?", (run_id,)
            )
            for a in answers:
                self.conn.execute(
                    """INSERT INTO frame_answers (
                           run_id, frame_index, timestamp_seconds, question_name,
                           question_instructions, criteria_json, top_choice, top_probability,
                           confidence, probabilities_json, raw_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        a.get("frame_index"),
                        a.get("timestamp_seconds"),
                        a.get("question_name"),
                        a.get("question_instructions"),
                        a.get("criteria_json"),
                        a.get("top_choice"),
                        a.get("top_probability"),
                        a.get("confidence"),
                        a.get("probabilities_json"),
                        a.get("raw_response"),
                    ),
                )
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM frame_answers WHERE run_id = ?", (run_id,)
        )
        return cur.fetchone()[0]

    def get_answers(
        self,
        run_id: int,
        question_name: Optional[str] = None,
        top_choice: Optional[str] = None,
        min_probability: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Get frame answers with optional filters."""
        query = (
            "SELECT fa.*, v.video_path AS source_video_path "
            "FROM frame_answers fa "
            "JOIN analysis_runs ar ON fa.run_id = ar.id "
            "JOIN videos v ON ar.video_id = v.id "
            "WHERE fa.run_id = ?"
        )
        params: List[Any] = [run_id]

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
        run_id: int,
        frame_index: int,
        question_name: str,
    ) -> Optional[Dict]:
        """Get a single answer by run, frame and question name."""
        cur = self.conn.execute(
            "SELECT * FROM frame_answers WHERE run_id = ? AND frame_index = ? AND question_name = ?",
            (run_id, frame_index, question_name),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_answers_for_video(
        self,
        video_id: int,
        run_id: Optional[int] = None,
    ) -> List[Dict]:
        """Get all frame answers for a video, optionally filtered by run."""
        query = (
            "SELECT fa.*, v.video_path AS source_video_path, ar.id as run_id, ar.params_hash "
            "FROM frame_answers fa "
            "JOIN analysis_runs ar ON fa.run_id = ar.id "
            "JOIN videos v ON ar.video_id = v.id "
            "WHERE ar.video_id = ?"
        )
        params: List[Any] = [video_id]
        if run_id is not None:
            query += " AND ar.id = ?"
            params.append(run_id)

        query += " ORDER BY fa.timestamp_seconds ASC"
        cur = self.conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

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

        # For backwards compatibility, find the run matching current params
        # (caller should pass the params_hash to look up specifically)
        answers = self.get_all_answers_for_video(video["id"])
        if not answers:
            return None, None

        results = answers
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
            # Delete old runs first (CASCADE handles frames when run deleted)
            self.conn.execute(
                "DELETE FROM analysis_runs WHERE updated_at < ?", (cutoff,)
            )
            self.conn.execute(
                "DELETE FROM cache_stats WHERE updated_at < ?", (cutoff,)
            )
            
    def get_invalid_videos(self) -> List[Tuple[str, str]]:
        """Get list of (video_path, reason) for invalid entries."""
        videos = self.conn.execute("SELECT video_path, content_hash FROM videos").fetchall()
        invalid = []
        for v in videos:
            path = v["video_path"]
            hash_val = v["content_hash"]
            if not os.path.exists(path):
                invalid.append((path, "File does not exist"))
            else:
                current_hash = _content_hash(path)
                if current_hash != hash_val:
                    invalid.append((path, f"Hash mismatch (expected {hash_val}, got {current_hash})"))
        return invalid

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
def init_cache(cache: str = "./.cache.db") -> CacheDB:
    """Initialize/cache database at the given file path."""
    return CacheDB(cache)