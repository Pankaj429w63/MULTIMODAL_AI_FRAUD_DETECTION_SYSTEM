"""
database.py — SQLite persistence layer for AI Fraud Detection System.
Stores every analysis run (scores, decision, reasons) and uploaded image blobs.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_detection.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # dict-like rows
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent access
    return conn


def init_db():
    """Create tables if they don't already exist."""
    conn = get_connection()
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS analysis_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            -- Transaction inputs
            amount              REAL,
            transaction_type    TEXT,
            old_balance         REAL,
            new_balance         REAL,
            -- NLP input
            complaint_text      TEXT,
            -- Scores
            transaction_score   REAL,
            complaint_score     REAL,
            identity_score      REAL,
            final_score         REAL,
            -- Decision
            decision            TEXT,
            -- Detailed reasons (JSON arrays)
            transaction_reasons TEXT,
            complaint_reasons   TEXT,
            identity_reasons    TEXT,
            risk_factors        TEXT,
            -- Image presence flags
            selfie_uploaded     INTEGER DEFAULT 0,
            id_uploaded         INTEGER DEFAULT 0,
            -- Context & Explainability
            location            TEXT,
            device_status       TEXT,
            time_of_day         INTEGER,
            account_age         INTEGER,
            xai_explanations    TEXT
        );

        CREATE TABLE IF NOT EXISTS uploads (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analysis_logs(id) ON DELETE CASCADE,
            file_type   TEXT    NOT NULL,   -- 'selfie' | 'id_card'
            filename    TEXT,
            file_data   BLOB    NOT NULL,
            timestamp   TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_logs_timestamp  ON analysis_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_decision   ON analysis_logs(decision);
        CREATE INDEX IF NOT EXISTS idx_uploads_analysis ON uploads(analysis_id);
        """)
        
        # Migrations for new fields (ignore expected errors if they already exist)
        for col in ["location TEXT", "device_status TEXT", "time_of_day INTEGER", "account_age INTEGER", "xai_explanations TEXT", "username TEXT"]:
            try:
                conn.execute(f"ALTER TABLE analysis_logs ADD COLUMN {col};")
            except sqlite3.OperationalError:
                pass
                
    conn.close()


# ─────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────

def save_analysis(
    amount: float,
    transaction_type: str,
    old_balance: float,
    new_balance: float,
    location: str,
    device_status: str,
    time_of_day: int,
    account_age: int,
    complaint_text: str,
    t_result: dict,
    c_result: dict,
    i_result: dict,
    fusion_result: dict,
    selfie_bytes: bytes | None = None,
    id_bytes:     bytes | None = None,
    selfie_name:  str  = "",
    id_name:      str  = "",
    username:     str  = "Guest",
) -> int:
    """
    Persist one complete analysis run.
    Returns the new analysis_log row id.
    """
    conn = get_connection()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with conn:
        cur = conn.execute(
            """
            INSERT INTO analysis_logs (
                timestamp, amount, transaction_type, old_balance, new_balance,
                complaint_text,
                transaction_score, complaint_score, identity_score, final_score,
                decision,
                transaction_reasons, complaint_reasons, identity_reasons, risk_factors,
                selfie_uploaded, id_uploaded,
                location, device_status, time_of_day, account_age, xai_explanations,
                username
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts, amount, transaction_type, old_balance, new_balance,
                complaint_text,
                t_result["score"], c_result["score"], i_result["score"],
                fusion_result["final_score"],
                fusion_result["decision"],
                json.dumps(t_result.get("reasons", [])),
                json.dumps(c_result.get("reasons", [])),
                json.dumps(i_result.get("reasons", [])),
                json.dumps(fusion_result.get("risk_factors", [])),
                1 if selfie_bytes else 0,
                1 if id_bytes     else 0,
                location, device_status, time_of_day, account_age,
                json.dumps(t_result.get("xai_explanations", {})),
                username
            ),
        )
        log_id = cur.lastrowid

        # Store image blobs
        if selfie_bytes:
            conn.execute(
                "INSERT INTO uploads (analysis_id, file_type, filename, file_data, timestamp) VALUES (?,?,?,?,?)",
                (log_id, "selfie", selfie_name, selfie_bytes, ts),
            )
        if id_bytes:
            conn.execute(
                "INSERT INTO uploads (analysis_id, file_type, filename, file_data, timestamp) VALUES (?,?,?,?,?)",
                (log_id, "id_card", id_name, id_bytes, ts),
            )

    conn.close()
    return log_id


# ─────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────

def fetch_logs(limit: int = 200, decision_filter: str = "All") -> list[dict]:
    """Return analysis logs, newest first."""
    conn = get_connection()
    if decision_filter == "All":
        rows = conn.execute(
            "SELECT * FROM analysis_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analysis_logs WHERE decision=? ORDER BY id DESC LIMIT ?",
            (decision_filter, limit),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_uploads_for(analysis_id: int) -> list[dict]:
    """Return image blobs for a given analysis run."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT file_type, filename, file_data FROM uploads WHERE analysis_id=?",
        (analysis_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def fetch_last_kyc(username: str) -> dict | None:
    """Retrieve the most recent successful KYC image blobs for a specific user."""
    if not username or username.strip() == "" or username.lower() == "guest": 
        return None
        
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM analysis_logs WHERE username=? AND selfie_uploaded=1 AND id_uploaded=1 ORDER BY id DESC LIMIT 1",
        (username,)
    ).fetchone()
    
    if not row:
        conn.close()
        return None
        
    analysis_id = row["id"]
    uploads = conn.execute(
        "SELECT file_type, filename, file_data FROM uploads WHERE analysis_id=?",
        (analysis_id,)
    ).fetchall()
    conn.close()
    
    result = {}
    for r in uploads:
        result[r["file_type"]] = (r["filename"], r["file_data"])
    
    if "selfie" in result and "id_card" in result:
        return result
    return None


def get_stats() -> dict:
    """Aggregate statistics for the dashboard."""
    conn = get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*)                                            AS total,
            SUM(CASE WHEN decision='FRAUD'  THEN 1 ELSE 0 END) AS fraud_count,
            SUM(CASE WHEN decision='POSSIBLY FRAUD' THEN 1 ELSE 0 END) AS pos_fraud_count,
            SUM(CASE WHEN decision='POSSIBLY LEGIT' THEN 1 ELSE 0 END) AS pos_legit_count,
            SUM(CASE WHEN decision='LEGIT'  THEN 1 ELSE 0 END) AS legit_count,
            ROUND(AVG(final_score), 4)                         AS avg_score,
            ROUND(AVG(transaction_score), 4)                   AS avg_tx,
            ROUND(AVG(complaint_score),   4)                   AS avg_nlp,
            ROUND(AVG(identity_score),    4)                   AS avg_kyc
        FROM analysis_logs
    """).fetchone()
    conn.close()
    return dict(row) if row else {}


def delete_log(log_id: int):
    """Delete a single analysis run (cascades to uploads)."""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM analysis_logs WHERE id=?", (log_id,))
    conn.close()


# Auto-initialize on import
init_db()
