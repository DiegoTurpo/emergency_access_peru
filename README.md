# Emergency Healthcare Access Inequality in Peru

Geospatial analysis of emergency healthcare access across 1,873 Peruvian districts, combining IPRESS facility data, SUSALUD emergency production records, and CCPP populated-center coordinates.

---

## Project Structure

```
emergency_access_peru/
├── app.py                      # Streamlit application (4 tabs)
├── requirements.txt
├── src/
│   ├── data_loader.py          # Raw file ingestion
│   ├── cleaning.py             # Cleaning pipeline + CleaningLog
│   ├── geospatial.py           # Task 2 — spatial joins and distance metrics
│   └── utils.py                # Encoding detection, UBIGEO padding
├── data/
│   ├── raw/                    # Source files (IPRESS.csv, DISTRITOS.shp, CCPP zip)
│   └── processed/              # Outputs: .gpkg and .parquet files
└── output/
    └── tables/
        └── cleaning_report.md  # Auditable cleaning decisions
```

---

## Data Sources

| Dataset | Description | Rows |
|---------|-------------|-----:|
| IPRESS | SUSALUD facility registry (all active establishments) | 20,793 |
| SUSALUD | Emergency care production 2015–2026 (Consulta-C1) | 2,193,587 |
| CCPP | INEI populated centers with coordinates | 136,543 |
| DISTRITOS | INEI district polygons (shapefile) | 1,873 |

---

## CRS Strategy

Two coordinate reference systems are used throughout the project:

### EPSG:4326 — WGS-84 (geographic, degrees)
- **Used for**: all stored data, GeoPackage outputs, Folium maps, and display.
- Every `.gpkg` file in `data/processed/` is in this CRS.
- Latitude/longitude in decimal degrees — universally understood and required by Folium.

### EPSG:24891 — PSAD56 / Peru Central Zone (projected, metres)
- **Used for**: metric calculations only — distances between CCPP and IPRESS, polygon areas for deduplication.
- This is Peru's official projected CRS; it minimises distance and area distortion for the Peruvian territory.
- Data is reprojected to EPSG:24891 for computation and immediately converted back to EPSG:4326 before saving.

**Rule of thumb**: all data at rest → EPSG:4326; any operation that requires metres → temporary reproject to EPSG:24891.

---

## Spatial Operations (Task 2)

### 1. Facility → District Assignment
Each IPRESS with valid GPS coordinates is assigned to a district polygon using `gpd.sjoin(..., predicate="within")`. A small number of facilities that fall on district borders or just outside polygons (due to coordinate precision) are assigned using `sjoin_nearest` as a fallback, after reprojecting both layers to EPSG:24891 for correct metric search.

### 2. Populated Center → District Assignment
Same two-step procedure applied to all 136,543 CCPP points. 221 points required the nearest-polygon fallback.

### 3. Nearest Emergency Facility Distance
For each populated center, `gpd.sjoin_nearest` is used to find the closest emergency-capable facility. Both layers are reprojected to EPSG:24891 so the `distance_col` output is in metres.

Two emergency facility definitions are computed in parallel:

| Definition | Criteria | Facilities |
|------------|----------|----------:|
| **Baseline** | Confirmed in SUSALUD emergency records + valid GPS | 3,093 |
| **Alternative** | Structural category ≥ I-3, regardless of SUSALUD reporting | 1,854 |

### 4. District-Level Metrics Aggregated
- **Supply**: `n_facilities_total`, `n_emergency_active`, `n_emergency_structural`, `camas_total`, `n_public`, `n_private`, `n_cat_I`, `n_cat_II_plus`
- **Access (baseline & alternative)**: `n_ccpp`, `dist_mean_m`, `dist_median_m`, `dist_max_m`, `pct_ccpp_gt5km`, `pct_ccpp_gt20km`
- **SUSALUD production**: latest-year `total_atenciones`, `total_atendidos`, `n_ipress_reportantes`

All metrics are stored in `district_master.gpkg` (1,873 rows × 28 columns).

---

## Key Findings (preliminary)

- **834 of 1,873 districts** (45%) have no SUSALUD-confirmed emergency facility within their boundaries.
- **National median distance** from a populated center to the nearest emergency facility: **6.5 km** (baseline).
- Worst-case distance reaches **341 km**, concentrated in remote Amazonian districts.

---

## Setup & Execution

### Environment

```bash
conda create -n homework2 python=3.11
conda install -n homework2 -c conda-forge \
    geopandas folium matplotlib seaborn pandas numpy scipy \
    pillow pyproj shapely fiona pyogrio openpyxl pyarrow chardet
pip install streamlit streamlit-folium plotly requests branca
```

### Run the pipeline

```bash
# Task 1 — Data cleaning
conda run -n homework2 python -c "
import sys; sys.path.insert(0,'.')
from src.cleaning import run_cleaning_pipeline
run_cleaning_pipeline()
"

# Task 2 — Geospatial integration
conda run -n homework2 python -c "
import sys; sys.path.insert(0,'.')
from src.geospatial import run_geospatial_pipeline
run_geospatial_pipeline()
"

# Streamlit app
conda run -n homework2 streamlit run app.py
```

---

## Cleaning Decisions

See [`output/tables/cleaning_report.md`](output/tables/cleaning_report.md) for a full record of every filtering step applied to each dataset, including row counts before and after each decision.
