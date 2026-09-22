"""
Minimal annotation backend for the building-footprint viewer.

Scope (deliberately minimal per the HW1 scaffold allowance):
- Does NOT serve the base GeoJSON — the frontend keeps loading that from
  the static file it already has. This backend only persists annotations
  (label/flag status + user-drawn footprints).
- One table, two routes: GET all annotations, POST to upsert one.

Run with:
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = "annotations.db"

app = FastAPI()

# CORS: your Vite dev server (localhost:5173) and this API (localhost:8000)
# are different origins as far as the browser is concerned, even though
# they're both "localhost". Without this, every fetch() from the frontend
# fails silently in the browser console with a CORS error, not a 404 —
# that's the #1 thing that eats time here if you skip it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                footprint_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'unchanged',
                note TEXT,
                source TEXT NOT NULL DEFAULT 'microsoft',
                geometry TEXT,
                updated_at TEXT
            )
            """
        )


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class Annotation(BaseModel):
    footprint_id: str | int
    status: str = "unchanged"  # unchanged | modified | demolished | new
    note: str | None = None
    source: str = "microsoft"  # microsoft | user_drawn
    geometry: str | None = None  # GeoJSON geometry as a JSON string; only
    # populated for source == "user_drawn", since Microsoft-sourced rows
    # already have their geometry in the static GeoJSON the frontend loads.


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/annotations")
def list_annotations():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM annotations").fetchall()
        return [dict(row) for row in rows]


@app.post("/annotations")
def upsert_annotation(annotation: Annotation):
    # INSERT OR REPLACE keyed on footprint_id: covers both "first time
    # annotating this footprint" and "updating an existing annotation"
    # with one code path, since the frontend always sends full current
    # state rather than a partial patch.
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO annotations
                (footprint_id, status, note, source, geometry, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                annotation.footprint_id,
                annotation.status,
                annotation.note,
                annotation.source,
                annotation.geometry,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    return {"ok": True, "footprint_id": annotation.footprint_id}