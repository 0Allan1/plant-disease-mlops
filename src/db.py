"""Tiny SQLite 'database' that stores every uploaded file and every retraining run.
It lives in the shared data volume, so all API containers see the same records."""
import sqlite3
import time
from contextlib import contextmanager

from src import config

STALE_RUN_SECONDS = 3 * 3600  # a 'running' row older than this is treated as crashed


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS uploads(
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, label TEXT, path TEXT,
            status TEXT DEFAULT 'pending', note TEXT, created_at REAL, processed_at REAL);
        CREATE TABLE IF NOT EXISTS retrain_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT, status TEXT, started_at REAL, finished_at REAL,
            n_new_images INTEGER, old_accuracy REAL, new_accuracy REAL, old_f1 REAL, new_f1 REAL,
            model_replaced INTEGER, message TEXT);
        """)


def add_upload(filename, label, path):
    with connect() as con:
        cur = con.execute("INSERT INTO uploads(filename,label,path,created_at) VALUES(?,?,?,?)",
                          (filename, label, str(path), time.time()))
        return cur.lastrowid


def pending_uploads():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY id")]


def mark_upload(upload_id, status, note=""):
    with connect() as con:
        con.execute("UPDATE uploads SET status=?, note=?, processed_at=? WHERE id=?",
                    (status, note, time.time(), upload_id))


def upload_counts():
    with connect() as con:
        rows = con.execute("SELECT label, status, COUNT(*) n FROM uploads GROUP BY label, status").fetchall()
    out = {"pending": 0, "processed": 0, "rejected": 0, "by_label": {}}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + r["n"]
        out["by_label"].setdefault(r["label"], {})[r["status"]] = r["n"]
    return out


def start_run(trigger):
    """Atomically create a 'running' row. Returns run id, or None if a run is already in progress."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        busy = con.execute("SELECT 1 FROM retrain_runs WHERE status='running' AND started_at>?",
                           (time.time() - STALE_RUN_SECONDS,)).fetchone()
        if busy:
            con.execute("COMMIT")
            return None
        cur = con.execute("INSERT INTO retrain_runs(trigger,status,started_at) VALUES(?,?,?)",
                          (trigger, "running", time.time()))
        con.execute("COMMIT")
        return cur.lastrowid


def finish_run(run_id, status, **fields):
    fields.update(status=status, finished_at=time.time())
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE retrain_runs SET {cols} WHERE id=?", (*fields.values(), run_id))


def latest_run():
    with connect() as con:
        r = con.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None


def list_runs(limit=10):
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,))]
