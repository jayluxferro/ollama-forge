"""Optional SQLite storage for eval run history (for plots over time and multi-model comparison)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def get_default_db_path() -> Path:
    return Path.home() / ".ollama_forge" / "security_eval_runs.db"


def init_db(db_path: str | Path | None = None) -> Path:
    """Create DB and runs table if not exists. Returns path used."""
    path = Path(db_path) if db_path else get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            base_url TEXT,
            prompt_set TEXT,
            timestamp_iso TEXT NOT NULL,
            kpis_json TEXT NOT NULL,
            results_json TEXT
        )
    """)
    conn.commit()
    conn.close()
    return path


def save_run(run_meta: dict, db_path: str | Path | None = None) -> None:
    """Append one run to the history DB."""
    path = Path(db_path) if db_path else get_default_db_path()
    init_db(path)
    kpis = run_meta.get("kpis") or {}
    conn = sqlite3.connect(path)
    conn.execute(
        """INSERT INTO runs (model, base_url, prompt_set, timestamp_iso, kpis_json, results_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            run_meta.get("model") or "",
            run_meta.get("base_url") or "",
            run_meta.get("prompt_set") or "",
            run_meta.get("timestamp_iso") or "",
            json.dumps(kpis),
            json.dumps(run_meta.get("results") or []),
        ),
    )
    conn.commit()
    conn.close()


def load_runs(db_path: str | Path | None = None, limit: int = 100) -> list[dict]:
    """Load recent runs: list of {id, model, base_url, prompt_set, timestamp_iso, kpis, results}."""
    path = Path(db_path) if db_path else get_default_db_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT id, model, base_url, prompt_set, timestamp_iso, kpis_json, results_json "
        "FROM runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "model": r[1],
                "base_url": r[2],
                "prompt_set": r[3],
                "timestamp_iso": r[4],
                "kpis": json.loads(r[5]) if r[5] else {},
                "results": json.loads(r[6]) if r[6] else [],
            }
        )
    return out
