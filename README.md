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

### Known limitation: footprint/imagery date mismatch

The building footprint polygons and the satellite basemap were **not captured at the same time**, and the imagery is the newer of the two:

- **Footprints:** per [Microsoft's own documentation](https://github.com/Microsoft/USBuildingFootprints/blob/master/README.md), most US footprints are from 2019–2020 imagery, with the remainder averaging ~2012. Our Albemarle features carry `release`/`capture_dates_range` properties consistent with this — the newer batch (`release: 2`) is dated `7/1/2018–8/1/2018`; the older batch (`release: 1`) has no recorded date.
- **Imagery:** the Esri World Imagery basemap is a live, continuously-updated composite, not a fixed vintage. We verified the actual source imagery for a sampled Albemarle coordinate via Esri's `identify` REST endpoint (`server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/identify`); the highest-draw-order (i.e. actually-rendered) source layer at that point was `Virginia2025`, dated **Feb–Mar 2025**.

**Net effect:** at the sampled location, satellite imagery is roughly **6–7 years newer** than the footprint polygon it's paired with. This is not necessarily uniform across the whole county (imagery vintage is composited from different source blocks region to region), but the imagery-newer-than-footprint direction is the one to assume. This is exactly what the annotation feature (below) is designed to let a user flag: a footprint that no longer matches what the current imagery shows.

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