import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class EventStore:
    def __init__(self, db_path: Path, pending_dir: Path):
        self.db_path     = db_path
        self.pending_dir = pending_dir
        pending_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id   TEXT NOT NULL,
                    type       TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'pending',
                    data       TEXT,
                    filepath   TEXT,
                    filename   TEXT,
                    created_at TEXT NOT NULL,
                    attempts   INTEGER DEFAULT 0,
                    priority   INTEGER NOT NULL DEFAULT 0
                )
            """)
            for col in (
                "event_id TEXT NOT NULL DEFAULT ''",
                "status TEXT NOT NULL DEFAULT 'pending'",
                "priority INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(f"ALTER TABLE events ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_event_type
                ON events (event_id, type)
            """)
            # reset any 'processing' rows left by a previous crash
            conn.execute("UPDATE events SET status = 'pending' WHERE status = 'processing'")

    def enqueue(self, type_: str, *, event_id: str, data: Optional[dict] = None,
                filepath: Optional[str] = None, filename: Optional[str] = None,
                priority: int = 0) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (event_id, type, status, data, filepath, filename, created_at, priority) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)",
                (
                    event_id,
                    type_,
                    json.dumps(data) if data is not None else None,
                    filepath,
                    filename,
                    datetime.now(timezone.utc).isoformat(),
                    priority,
                ),
            )

    def next_pending(self) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE status = 'pending' ORDER BY priority DESC, id ASC LIMIT 1"
            ).fetchone()
            if row is not None:
                conn.execute("UPDATE events SET status = 'processing' WHERE id = ?", (row['id'],))
            return row

    def mark_done(self, id_: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM events WHERE id = ?", (id_,))

    def increment_attempts(self, id_: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE events SET attempts = attempts + 1, status = 'pending' WHERE id = ?",
                (id_,),
            )

    def add_to_data(self, event_id: str, type_: str, extra: dict) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM events WHERE event_id = ? AND type = ?",
                (event_id, type_),
            ).fetchone()
            if row:
                data = json.loads(row['data']) if row['data'] else {}
                data.update(extra)
                conn.execute(
                    "UPDATE events SET data = ? WHERE event_id = ? AND type = ?",
                    (json.dumps(data), event_id, type_),
                )

    def has_pending_sibling(self, event_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM events WHERE event_id = ? AND type IN ('photo','video') AND status IN ('pending','processing') LIMIT 1",
                (event_id,),
            ).fetchone()
            return row is not None

    def count_pending(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM events WHERE status != 'done'").fetchone()[0]
