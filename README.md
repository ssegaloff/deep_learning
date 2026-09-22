# Albemarle County Building Footprint Viewer

(Please note this README.md was written with Claude.)

## Description
A web-based viewer for exploring building footprint data over Albemarle County, VA. Users can click any building on the map to review it against current satellite imagery and record whether it matches, has changed, or is no longer present — laying the groundwork for later assignments that will use deep learning to detect these kinds of changes automatically from imagery.

## Data
- **Type:** Geospatial vector data (building footprint polygons, GeoJSON)
- **Source:** Microsoft US Building Footprints (ODbL license), clipped to Albemarle County, VA via a `ogr2ogr`-based pipeline (see `data-pipeline/`)
- Each footprint is assigned a stable `footprint_id` at pipeline time, since the source data has no persistent unique ID of its own

## Features Implemented
- Interactive Leaflet map of Albemarle County with satellite imagery basemap
- Building footprints rendered as clickable GeoJSON polygons
- Clicking a building opens a popup showing its data-release/capture-date metadata
- **Annotation:** the popup lets the user label a footprint's current status (`unchanged` / `modified` / `demolished`) and add a free-text note
- **Backend:** annotations are persisted to a SQLite database via a FastAPI backend, so labels survive a page refresh

## Annotation: Yes
Implemented as a label/flag workflow — clicking a building opens a form to set a status and note, which is saved via the backend (see below).

## Backend/Database: Yes
- **Stack:** FastAPI + SQLite
- **Scope:** minimal scaffold, storing annotations only (footprint geometry itself is still served statically from the frontend's bundled GeoJSON)
- **Endpoints:**
  - `GET /annotations` — returns all saved annotations
  - `POST /annotations` — upserts one annotation, keyed by `footprint_id`

## Libraries / Frameworks
- **Frontend:** React (Vite), react-leaflet, Leaflet
- **Backend:** FastAPI, SQLite (via Python's built-in `sqlite3`), Pydantic, Uvicorn
- **Data pipeline:** GDAL / `ogr2ogr`

## Running the Application

### 1. Frontend
```bash
cd geo-data-viewer
npm install
npm run dev
```
Opens at `http://localhost:5173`.

### 2. Backend
From the repo root, in a separate terminal:
```bash
source venv/bin/activate
uv pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```
Runs at `http://localhost:8000`. The frontend must be running at `http://localhost:5173` for CORS to allow requests between the two (configured in `backend/main.py`).

Both servers need to be running simultaneously for annotations to load and save.