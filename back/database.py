import sqlite3
import time
from contextlib import contextmanager

DB_PATH = "./messages.db"


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                label TEXT,
                created_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_conversation(conversation_id: str, label: str = ""):
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, label, created_at) VALUES (?, ?, ?)",
            (conversation_id, label, time.time()),
        )


def save_message(conversation_id: str, role: str, content: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, time.time()),
        )


def get_messages(conversation_id: str):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversations():
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, label, created_at FROM conversations ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_conversation(conversation_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
