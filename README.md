# Albemarle County Building Footprint Viewer

DS7400 (Deep Learning, Prof. Baek) — HW1: Build the Foundation for Your Deep Learning Application

## Description

<!-- TODO: 2-3 sentences on what this app does, once the viewer is fully locked -->
An interactive web-based viewer for building footprint polygons over satellite imagery, focused on Albemarle County, VA. Built as the foundation for a semester-long deep learning application; future homeworks will add DL capability on top of this viewer (e.g. classification, detection, change detection).

## Data

- **Source:** [Microsoft US Building Footprints](https://github.com/microsoft/USBuildingFootprints) (ODbL license), Virginia state file.
- **Processing:** Clipped to an Albemarle County bounding box (sourced from Nominatim) using `ogr2ogr`, then each feature is assigned a `footprint_id` and written to `geo-data-viewer/public/albemarle-buildings.geojson`.
- **CRS:** EPSG:4326 (WGS84).
- **Feature count:** ~77,897 building footprints after clipping.
- **Note on `footprint_id`:** assigned as a sequential index on each pipeline run — not guaranteed stable across reruns if the source data changes. Flagging in case later homeworks build annotations keyed on this id.

The raw Virginia source file (~800MB) is **not committed to this repo** — it's large and not something that belongs in version control. See [Running the data pipeline](#running-the-data-pipeline) below to regenerate it locally.

## Running the Application

<!-- TODO: fill in once viewer is finalized -->
```bash
cd geo-data-viewer
npm install
npm run dev
```
Then open `http://localhost:5173` (or whatever port Vite reports).

## Running the Data Pipeline

Only needed if you want to regenerate `albemarle-buildings.geojson` from source (it's already committed in `geo-data-viewer/public/`, so this is optional for just running the viewer).

**System dependency:** GDAL (provides `ogr2ogr`).
```bash
brew install gdal      # macOS, via Homebrew
ogr2ogr --version      # verify install
```

**Python dependencies:** managed via `uv`.
```bash
uv sync
```

**Data:** download the Virginia GeoJSON from [Microsoft US Building Footprints](https://github.com/microsoft/USBuildingFootprints) and place it at `~/gis-scratch/Virginia.geojson`.

**Run:**
<!-- TODO: confirm exact invocation / working directory -->
```bash
uv run python data-pipeline/build_buildings_data.py
```
This regenerates `geo-data-viewer/public/albemarle-buildings.geojson`. The script is idempotent — safe to rerun any number of times.

## Major Libraries / Frameworks

<!-- TODO: confirm full list once viewer is finalized -->
- **Frontend:** React (Vite), react-leaflet (map rendering), Esri World Imagery (satellite basemap tiles)
- **Data pipeline:** GDAL/`ogr2ogr` (clipping), `geopandas` (id assignment, GeoJSON I/O)

## Features Implemented

<!-- TODO: finalize this list against actual submitted state -->
- Satellite imagery basemap (Esri World Imagery)
- Building footprint polygons rendered as an interactive layer (Leaflet `GeoJSON`, canvas renderer for performance at ~78k features)
- Click-to-popup on individual buildings, showing footprint metadata (`release`, `capture_dates_range`)
- [ ] Zoom / pan — confirm status
- [ ] Filename/metadata display — confirm what's shown beyond popup

## Annotation

<!-- TODO -->
Not yet implemented. <!-- update if pursued for +10 -->

## Backend / Database

<!-- TODO -->
Not yet implemented. <!-- update if pursued for +10 --># Albemarle County Building Footprint Viewer

DS7400 (Deep Learning, Prof. Baek) — HW1: Build the Foundation for Your Deep Learning Application

## Description

<!-- TODO: 2-3 sentences on what this app does, once the viewer is fully locked -->
An interactive web-based viewer for building footprint polygons over satellite imagery, focused on Albemarle County, VA. Built as the foundation for a semester-long deep learning application; future homeworks will add DL capability on top of this viewer (e.g. classification, detection, change detection).

## Data

- **Source:** [Microsoft US Building Footprints](https://github.com/microsoft/USBuildingFootprints) (ODbL license), Virginia state file.
- **Processing:** Clipped to an Albemarle County bounding box (sourced from Nominatim) using `ogr2ogr`, then each feature is assigned a `footprint_id` and written to `geo-data-viewer/public/albemarle-buildings.geojson`.
- **CRS:** EPSG:4326 (WGS84).
- **Feature count:** ~77,897 building footprints after clipping.
- **Note on `footprint_id`:** assigned as a sequential index on each pipeline run — not guaranteed stable across reruns if the source data changes. Flagging in case later homeworks build annotations keyed on this id.

The raw Virginia source file (~800MB) is **not committed to this repo** — it's large and not something that belongs in version control. See [Running the data pipeline](#running-the-data-pipeline) below to regenerate it locally.

## Running the Application

<!-- TODO: fill in once viewer is finalized -->
```bash
cd geo-data-viewer
npm install
npm run dev
```
Then open `http://localhost:5173` (or whatever port Vite reports).

## Running the Data Pipeline

Only needed if you want to regenerate `albemarle-buildings.geojson` from source (it's already committed in `geo-data-viewer/public/`, so this is optional for just running the viewer).

**System dependency:** GDAL (provides `ogr2ogr`).
```bash
brew install gdal      # macOS, via Homebrew
ogr2ogr --version      # verify install
```

**Python dependencies:** managed via `uv`.
```bash
uv sync
```

**Data:** download the Virginia GeoJSON from [Microsoft US Building Footprints](https://github.com/microsoft/USBuildingFootprints) and place it at `~/gis-scratch/Virginia.geojson`.

**Run:**
<!-- TODO: confirm exact invocation / working directory -->
```bash
uv run python data-pipeline/build_buildings_data.py
```
This regenerates `geo-data-viewer/public/albemarle-buildings.geojson`. The script is idempotent — safe to rerun any number of times.

## Major Libraries / Frameworks

<!-- TODO: confirm full list once viewer is finalized -->
- **Frontend:** React (Vite), react-leaflet (map rendering), Esri World Imagery (satellite basemap tiles)
- **Data pipeline:** GDAL/`ogr2ogr` (clipping), `geopandas` (id assignment, GeoJSON I/O)

## Features Implemented

<!-- TODO: finalize this list against actual submitted state -->
- Satellite imagery basemap (Esri World Imagery)
- Building footprint polygons rendered as an interactive layer (Leaflet `GeoJSON`, canvas renderer for performance at ~78k features)
- Click-to-popup on individual buildings, showing footprint metadata (`release`, `capture_dates_range`)
- [ ] Zoom / pan — confirm status
- [ ] Filename/metadata display — confirm what's shown beyond popup

## Annotation

<!-- TODO -->
Not yet implemented. <!-- update if pursued for +10 -->

## Backend / Database

<!-- TODO -->
Not yet implemented. <!-- update if pursued for +10 -->