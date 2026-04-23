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

Each task writes its outputs to `data/processed/` or `output/`; run them in order.

```bash
# Task 1 — Data cleaning  →  data/processed/*.gpkg + *.parquet
conda run -n homework2 python -m src.cleaning

# Task 2 — Geospatial integration  →  district_master.gpkg, ccpp_with_distances.gpkg
conda run -n homework2 python -m src.geospatial

# Task 3 — HADI metrics  →  district_master.gpkg (enriched), district_hadi.csv
conda run -n homework2 python -m src.metrics

# Task 4 — Static figures  →  output/figures/fig01_*.png … fig06_*.png
conda run -n homework2 python -m src.visualization

# Streamlit app (Tasks 5 & 6)
conda run -n homework2 streamlit run app.py
```

---

## Static Visualizations (Task 4)

Six figures in `output/figures/`, each answering one or more analytical questions:

---

### fig01 — Emergency facility supply distribution
**Answers:** Q1 (facility availability across districts)

Histogram of SUSALUD-confirmed emergency facilities per district, on a log y-scale.

**Why this chart:** The distributional *shape* is the finding — an extreme right skew with a dominant spike at zero (834 districts, 44.5%, have no emergency facility). A log y-scale keeps both the mass at zero and the long tail visible simultaneously. **Why not a bar chart of top districts:** ranking individual districts buries the dominant story (near-half with zero); the histogram makes the inequality structural rather than anecdotal.

---

### fig02 — Facility count vs emergency care activity
**Answers:** Q1 (both the facility and activity sub-dimensions, and their relationship)

Scatter of log(n\_emergency\_active + 1) vs log(SUSALUD emergency visits + 1), coloured by HADI quintile, with an OLS trend line.

**Why this chart:** A scatter is the only chart type that simultaneously shows *both* Q1 dimensions and reveals the relationship between them. The log1p transform compresses the 0–262k range while keeping zero-valued districts at the origin (a visible cluster, not hidden by the axis). The OLS slope (2.6) confirms a positive association but the wide vertical scatter shows that many districts with few facilities still record high activity — and vice versa — exposing SUSALUD reporting gaps. **Why not two separate histograms:** they lose the joint relationship and the quintile colouring that links this chart back to Task 3.

---

### fig03 — Spatial access: national distribution + geographic pattern
**Answers:** Q2 (which districts have weaker spatial access to emergency services)

Two panels: (A) histogram of % populated centres more than 20 km from the nearest facility; (B) horizontal box plots of median CCPP distance by department, sorted by median.

**Why this chart:** Panel A shows the national distribution shape — zero-inflated, with 1,188 perfectly well-served districts and 125 districts where *all* populated centres exceed 20 km. Panel A alone cannot say *where* the isolation is concentrated; Panel B adds the geographic dimension without needing a map (Task 5). Together they answer Q2 more fully than either alone. **Why not a per-district sorted bar:** 1,873 bars are unreadable; the department grouping reveals the Amazonian cluster (Loreto, Madre de Dios, Ucayali) as the structural geographic driver.

---

### fig04 — HADI score distribution: baseline vs alternative
**Answers:** Q3 (overall deprivation spectrum) + Q4 (how robust is the index?)

Histogram of baseline HADI with quintile bands as background shading; overlaid KDE curves for baseline (solid) and alternative (dashed).

**Why this chart:** The dual KDE is the most direct way to compare two continuous distributions on the same scale. The bands make the quintile membership visually immediate. The notable spike near 0.6 corresponds to the large cluster of districts that have zero facilities and zero SUSALUD activity but whose populated centres happen to lie within 20 km of a neighbouring district's facility — a finding about *structural* vs *spatial* underservice. The alternative KDE shifts rightward above 0.6, confirming that the stricter facility definition (1,854 vs 3,093) reclassifies a subset of districts as more deprived. **Why not two separate histograms:** comparing peaks and tails across two panels requires more cognitive work than a single overlay.

---

### fig05 — HADI components by quintile
**Answers:** Q3 (methodology — what drives the classification?)

Grouped bar chart: mean component score (Facility Density, Emergency Activity, Spatial Access) for each HADI quintile.

**Why this chart:** A grouped bar chart is the clearest way to read absolute values across an ordinal axis with three categories. The key finding — that spatial access (green) is *lower* in Q3 than facility and activity components, but overtakes them in Q4–Q5 — is immediately visible and would be hidden in a radar chart (distorted area perception) or a heatmap (requires colour interpretation for numeric comparison). The 0.5 reference line (national average by construction) contextualises each bar. **Why not a radar/spider chart:** radar distorts relative magnitudes and is harder to read for five groups simultaneously.

---

### fig06 — Sensitivity: baseline vs alternative HADI
**Answers:** Q4 (comparison between specifications)

Scatter of HADI baseline (x) vs HADI alternative (y), coloured by quintile shift (-3 to +3), with a y = x reference diagonal.

**Why this chart:** A scatter on the same scale with a y = x reference is the most direct way to visualise agreement vs disagreement between two continuous scores. Points on the diagonal are unchanged; points above it are reclassified as more deprived under the alternative. Colouring by the magnitude of shift adds the quintile dimension without requiring a separate confusion matrix. The result — most districts cluster tightly on the diagonal, but a visible off-diagonal cloud shifts upward (555 districts worsened by 1 quintile) — quantifies the sensitivity as modest but non-negligible. **Why not a cross-tab alone:** a confusion matrix of 5×5 quintile transitions loses the continuous HADI information and hides which specific score ranges disagree most.

---

## Cleaning Decisions

See [`output/tables/cleaning_report.md`](output/tables/cleaning_report.md) for a full record of every filtering step applied to each dataset, including row counts before and after each decision.
