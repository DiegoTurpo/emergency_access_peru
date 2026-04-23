# Emergency Healthcare Access Inequality in Peru

Geospatial analysis of emergency healthcare access across 1,873 Peruvian districts, combining IPRESS facility data, SUSALUD emergency production records, and CCPP populated-center coordinates.

---

## What Does This Project Do?

This project builds a **complete geospatial analytics pipeline** in Python to study emergency healthcare access inequality across all districts in Peru. It ingests four public datasets, cleans and integrates them spatially, constructs a composite deprivation index (HADI), produces static and interactive visualizations, and presents the full analysis in a Streamlit web application.

---

## Main Analytical Goal

> **Which districts in Peru appear relatively better or worse served in emergency healthcare access, and what evidence supports that conclusion?**

The project answers four specific research questions:

| # | Question |
|---|----------|
| **Q1** | Which districts have lower or higher availability of health facilities and emergency care activity? |
| **Q2** | Which districts have populated centers with weaker spatial access to emergency services? |
| **Q3** | Which districts are most and least underserved when combining all dimensions? |
| **Q4** | How sensitive are the results to the choice of facility definition? |

---

## Project Structure

```
emergency_access_peru/
├── app.py                          # Streamlit application (4 tabs)
├── requirements.txt                # Python dependencies
├── README.md
│
├── src/
│   ├── data_loader.py              # Raw file ingestion (one loader per dataset)
│   ├── cleaning.py                 # Cleaning pipeline + CleaningLog
│   ├── geospatial.py               # Spatial joins, distance metrics, district_master
│   ├── metrics.py                  # HADI index construction (Task 3)
│   ├── visualization.py            # Static charts (Task 4) + maps (Task 5)
│   └── utils.py                    # Encoding detection, UBIGEO padding
│
├── data/
│   ├── raw/                        # Source files (IPRESS.csv, DISTRITOS.shp, CCPP zip)
│   └── processed/                  # Outputs: .gpkg and .parquet files (git-ignored)
│
├── output/
│   ├── figures/                    # Static PNGs (fig01–fig06, map01–map03) + Folium HTML
│   └── tables/                     # district_hadi.csv, cleaning_report.md, metrics_report.md
│
└── video/
    └── link.txt                    # Link to explanatory video
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

## How Were the District-Level Metrics Constructed?

The district-level analytical output is the **HADI — Healthcare Access Deprivation Index** (Task 3, `src/metrics.py`).

### Index Design

HADI ∈ [0, 1]; **higher score = more deprived**. It combines three equal-weight components:

| Component | Variable | Deprivation direction |
|-----------|----------|-----------------------|
| **C1 — Facility Density** | Emergency facilities per 100 populated centers | Fewer facilities → higher deprivation |
| **C2 — Emergency Activity** | SUSALUD emergency visits per populated center | Fewer visits → higher deprivation |
| **C3 — Spatial Access** | % populated centers > 20 km from nearest emergency facility | Higher isolation → higher deprivation |

### Normalization

Each component is **percentile-ranked** and rescaled to [0, 1]:

```
rank_i = (rank(x_i) - 1) / (n - 1)
```

Percentile rank was chosen over min-max scaling because the activity distribution is extremely right-skewed (median = 0, max = 262,245 visits). Min-max would compress 99% of districts into a tiny band near zero.

### Composite Score

```
HADI = (C1 + C2 + C3) / 3
```

### Quintile Classification

Districts are classified into five quintiles using **equal-interval bins** [0, 0.2, 0.4, 0.6, 0.8, 1.0]:

| Quintile | HADI range | Interpretation |
|----------|------------|----------------|
| Q1 (Best) | 0.0 – 0.2 | Best-served districts |
| Q2        | 0.2 – 0.4 | Above-average access |
| Q3        | 0.4 – 0.6 | Moderate deprivation |
| Q4        | 0.6 – 0.8 | High deprivation |
| Q5 (Worst)| 0.8 – 1.0 | Most deprived districts |

Equal-interval bins were chosen over equal-frequency (`qcut`) because many districts tie at the same HADI score (e.g., all districts with zero facilities and zero activity), which causes `qcut` to produce highly unequal quintiles.

### Two Specifications (Q4)

| Specification | Facility definition | n facilities |
|---------------|---------------------|--------------|
| **Baseline** | Confirmed in SUSALUD emergency records + valid GPS | 3,093 |
| **Alternative** | Structural category ≥ I-3, regardless of SUSALUD reporting | 1,854 |

The baseline captures **observed** emergency activity; the alternative captures **structural capacity** regardless of reporting compliance. Both are computed in parallel and compared via a quintile-shift analysis.

---

## Key Findings

**Q1 — Territorial availability**
- **834 of 1,873 districts** (44.5%) have no SUSALUD-confirmed emergency facility within their boundaries.
- The most facility-rich district is **Cutervo (Cajamarca)** with 47 emergency facilities.
- The highest emergency care activity is in **Ica** with 262,245 SUSALUD visits — despite having only 5 confirmed facilities, revealing a major reporting gap.

**Q2 — Settlement access**
- **National median distance** from a populated center to the nearest emergency facility: **6.5 km** (baseline).
- **125 districts** (7%) have 100% of their populated centers beyond 20 km from any facility.
- The most isolated district is **Purus (Ucayali)** with a median distance of **231.8 km** to the nearest facility.
- Loreto, Madre de Dios, and Ucayali account for the bulk of spatial isolation.

**Q3 — Overall deprivation (HADI)**
- **134 districts** fall in Q5 (most deprived), concentrated in Loreto, Ucayali, and Amazonas.
- **150 districts** fall in Q1 (best served), concentrated in Lima, Callao, and Arequipa metropolitan areas.
- The worst-scoring districts (HADI ≈ 0.813): Tigre, Purus, Yaguas, Alto Nanay, Yavari — all remote Amazonian districts with zero facilities and 100% of CCPP beyond 20 km.
- The best-scoring district: **Lima** (HADI = 0.106).

**Q4 — Methodological sensitivity**
- **848 districts** (45%) are unchanged between baseline and alternative definitions.
- **555 districts** (30%) worsen by 1 quintile under the alternative — the stricter structural definition excludes SUSALUD-reporting facilities.
- **5 districts** shift by 3 quintiles (e.g., Cahuacho, Arequipa: Q2 → Q5).
- The alternative definition is most impactful for districts near the Q2/Q3 boundary that relied on facilities not reporting to SUSALUD.

---

## Setup & Execution

### How to Install the Dependencies

```bash
conda create -n homework2 python=3.11
conda install -n homework2 -c conda-forge \
    geopandas folium matplotlib seaborn pandas numpy scipy \
    pillow pyproj shapely fiona pyogrio openpyxl pyarrow chardet
pip install streamlit streamlit-folium plotly requests branca
```

### How to Run the Processing Pipeline

Each task writes its outputs to `data/processed/` or `output/`; run them in order:

```bash
# Task 1 — Data cleaning  →  data/processed/*.gpkg + *.parquet
conda run -n homework2 python -m src.cleaning

# Task 2 — Geospatial integration  →  district_master.gpkg, ccpp_with_distances.gpkg
conda run -n homework2 python -m src.geospatial

# Task 3 — HADI metrics  →  district_master.gpkg (enriched), district_hadi.csv
conda run -n homework2 python -m src.metrics

# Task 4 & 5 — Static figures and maps  →  output/figures/
conda run -n homework2 python -m src.visualization
```

### How to Run the Streamlit App

```bash
conda run -n homework2 streamlit run app.py
```

The app opens at **http://localhost:8501** and contains 4 tabs:
- **Tab 1 — Data & Methodology**: problem statement, data sources, cleaning decisions, HADI methodology, limitations.
- **Tab 2 — Static Analysis**: 6 figures answering Q1–Q4, with inline key findings and named district rankings.
- **Tab 3 — GeoSpatial Results**: 3 static maps + department-level summary table and Q5 bar chart.
- **Tab 4 — Interactive Exploration**: Folium maps (hover any district), district search/filter table, baseline vs alternative scatter, HADI component chart.

---

## Main Limitations

- **Reporting gap**: SUSALUD data only covers facilities that voluntarily submitted records. The 834 districts with zero confirmed facilities may still have informal emergency services that are not captured.
- **Static snapshot**: metrics use the latest available SUSALUD year per district, which varies from 2018 to 2026 depending on when each district last reported.
- **Euclidean distance**: straight-line distances from populated centers to facilities do not account for road network quality, river-crossing requirements, or terrain barriers — a critical limitation in Loreto, where river access is the only viable route.
- **CCPP as population proxy**: populated centers are counted equally regardless of actual population size. A large city and a hamlet of 10 people are treated identically in the access metrics.
- **Structural facility definition**: the alternative definition uses registered category (≥ I-3) as a proxy for emergency capability, but registration does not guarantee that a facility is actually operational or adequately staffed.

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

## Geospatial Maps (Task 5)

Five map outputs in `output/figures/` — three static (GeoPandas/matplotlib) and two interactive (Folium):

---

### map01 — HADI Quintile Choropleth *(static)*
**Answers:** Q3 — Which districts are most underserved overall?

Full-Peru choropleth of HADI baseline quintile (Q1=blue → Q5=red) with white department boundary overlay and a legend showing district counts per quintile.

**Why this chart:** A choropleth is the only type that simultaneously answers WHICH districts are deprived AND WHERE they are. The map reveals a clear coastal–Amazonian divide: Q1 districts (best served) cluster along the Pacific coast and in Lima; Q5 districts concentrate in Loreto, Ucayali, and Amazonas — a geographic pattern invisible in any statistical chart. **Alternative rejected:** dot density map adds visual noise without revealing the polygon-level administrative structure used in resource allocation decisions.

---

### map02 — Baseline vs Alternative Side-by-Side *(static)*
**Answers:** Q4 — Where does the facility definition change the classification?

Two panels at identical scale: left = baseline (3,093 SUSALUD-confirmed facilities), right = alternative (1,854 structural). Same colour scale enables direct visual comparison.

**Why this chart:** A side-by-side map externalises the cognitive comparison — the reader scans for colour changes without needing to subtract values mentally. The alternative panel shows a notable darkening of the Amazon interior (fewer qualifying facilities → higher deprivation) and some lightening of the highlands (structural category captures facilities not in SUSALUD). **Alternative rejected:** a single "difference" map (showing quintile shift) loses the absolute deprivation level, which is the primary policy-relevant quantity.

---

### map03 — Spatial Access Gap *(static)*
**Answers:** Q2 — Which districts have populated centers far from emergency care?

Continuous choropleth of `pct_ccpp_gt20km_baseline` (% of populated centers beyond 20 km from the nearest emergency facility). White/green = well-covered; red = fully isolated.

**Why this chart:** A continuous graduated scale preserves the full gradient of spatial isolation and avoids arbitrary discretisation. The access gap map reveals a distinct spatial pattern from the HADI composite: several Q3 districts (moderate overall deprivation) show deep red — they have facilities nearby in aggregate but their populated centers are scattered far from them. This decoupling of supply shortage and spatial isolation is only visible by mapping the components separately. **Alternative rejected:** using only the HADI map (map01) conflates the access component with the supply and activity components.

---

### map_hadi_explorer — Interactive HADI explorer *(Folium)*
**Answers:** Q3 + Q4 interactively

- **Layer 1 (default):** Baseline HADI quintile choropleth — hover any district to see UBIGEO, quintile, HADI score, emergency facility count, % CCPP > 20 km, median distance, and SUSALUD visits.
- **Layer 2 (toggle):** Alternative HADI quintile — same tooltip for the alternative specification.
- **Layer control** (top-right ≡): switch between specifications to observe reclassification in geographic context.

This map lets the Streamlit reader drill into any district of interest; static maps cannot show six simultaneous metrics per district.

---

### map_facilities_access — Facility locations on access background *(Folium)*
**Answers:** Q1 + Q2 interactively

- **Background:** `pct_ccpp_gt20km_baseline` choropleth (green = well-covered, red = isolated).
- **Blue markers (default on):** 3,093 SUSALUD-confirmed emergency facilities — hover for name, category, institution type.
- **Orange markers (toggle):** Structural-only facilities (category ≥ I-3 but NOT in SUSALUD data) — shows the additional facilities the alternative definition would include.

The visual conjunction of red districts with no blue markers identifies the hardest-to-serve areas — where spatial isolation AND facility absence compound each other.

---

## Cleaning Decisions

See [`output/tables/cleaning_report.md`](output/tables/cleaning_report.md) for a full record of every filtering step applied to each dataset, including row counts before and after each decision.
