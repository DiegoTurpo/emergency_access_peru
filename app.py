"""
Emergency Healthcare Access Inequality in Peru
Streamlit application — 4 tabs as required by the assignment.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "output" / "figures"
TABLES_DIR  = ROOT / "output" / "tables"
PROCESSED   = ROOT / "data" / "processed"

# ── Constants ─────────────────────────────────────────────────────────────────

DEPT_NAMES = {
    1: "Amazonas",      2: "Áncash",        3: "Apurímac",
    4: "Arequipa",      5: "Ayacucho",       6: "Cajamarca",
    7: "Callao",        8: "Cusco",          9: "Huancavelica",
    10: "Huánuco",      11: "Ica",           12: "Junín",
    13: "La Libertad",  14: "Lambayeque",    15: "Lima",
    16: "Loreto",       17: "Madre de Dios", 18: "Moquegua",
    19: "Pasco",        20: "Piura",         21: "Puno",
    22: "San Martín",   23: "Tacna",         24: "Tumbes",
    25: "Ucayali",
}

QUINTILE_COLORS = {
    "Q1 (Best)":  "#2c7bb6",
    "Q2":         "#abd9e9",
    "Q3":         "#ffffbf",
    "Q4":         "#fdae61",
    "Q5 (Worst)": "#d7191c",
}

QUINTILE_ORDER = ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"]

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_hadi() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "district_hadi.csv", dtype={"ubigeo": str})
    df["dept_name"] = df["iddpto"].map(DEPT_NAMES).fillna(df["iddpto"].astype(str))
    df["quintile_shift"] = pd.to_numeric(df["quintile_shift"], errors="coerce")
    # Build a readable label: "Distrito, Provincia (Departamento)"
    if "distrito" in df.columns:
        df["district_label"] = (
            df["distrito"].str.title() + ", " +
            df["provincia"].str.title() + " (" +
            df["departamento"].str.title() + ")"
        )
    else:
        df["district_label"] = df["ubigeo"]
    return df


@st.cache_data
def load_markdown(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"*(file not found: {path.name})*"


@st.cache_data
def load_folium_html(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<p>Map file not found — run the pipeline first.</p>"


# ── App config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emergency Healthcare Access — Peru",
    page_icon="🏥",
    layout="wide",
)

st.title("Emergency Healthcare Access Inequality in Peru")
st.caption(
    "Geospatial analysis of 1,873 districts · IPRESS facility registry · "
    "SUSALUD emergency records 2015–2026 · INEI populated centers"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ GeoSpatial Results",
    "🔍 Interactive Exploration",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & Methodology
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Data & Methodology")

    # ── Problem statement ──────────────────────────────────────────────────────
    st.subheader("Problem Statement")
    st.markdown(
        """
        Emergency healthcare access in Peru is highly unequal. Peru has 1,873 administrative
        districts spanning coastal deserts, Andean highlands, and Amazonian rainforest.
        This project quantifies **where** emergency services are scarce and **how far** populations
        must travel to reach them, combining three national datasets into a single district-level
        Healthcare Access Deprivation Index (HADI).

        **Research questions**

        | # | Question |
        |---|----------|
        | Q1 | How are emergency facilities and care activity distributed across districts? |
        | Q2 | Which districts have populated centers far from the nearest emergency facility? |
        | Q3 | What is the overall spectrum of emergency healthcare deprivation (HADI)? |
        | Q4 | How sensitive are the results to the choice of facility definition? |
        """
    )

    # ── Data sources ──────────────────────────────────────────────────────────
    st.subheader("Data Sources")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            | Dataset | Description | Rows |
            |---------|-------------|-----:|
            | **IPRESS** | SUSALUD facility registry (all active) | 20,793 |
            | **SUSALUD** | Emergency care production 2015–2026 | 2,193,587 |
            | **CCPP** | INEI populated centers with coordinates | 136,543 |
            | **DISTRITOS** | INEI district polygons (shapefile) | 1,873 |
            """
        )
    with col_b:
        st.markdown(
            """
            **Two facility definitions are used in parallel:**

            | Definition | Criteria | n |
            |------------|----------|--:|
            | **Baseline** | Confirmed in SUSALUD emergency records + valid GPS | 3,093 |
            | **Alternative** | Structural category ≥ I-3, regardless of SUSALUD reporting | 1,854 |

            The baseline definition captures **observed** emergency activity; the alternative
            captures **structural capacity** regardless of reporting compliance.
            """
        )

    # ── Cleaning decisions ────────────────────────────────────────────────────
    st.subheader("Cleaning & Filtering Decisions")
    cleaning_md = load_markdown(TABLES_DIR / "cleaning_report.md")
    with st.expander("Show full cleaning report", expanded=False):
        st.markdown(cleaning_md)
    st.markdown(
        """
        Key decisions at a glance:
        - **IPRESS**: removed 26 duplicates; kept only `estado == ACTIVADO`.
        - **SUSALUD**: dropped 237,753 non-reporter rows (`NE_0001/NE_0002`) and 502,040 exact duplicates.
        - **CCPP**: dropped 44 coordinate-duplicate populated centers.
        """
    )

    # ── CRS strategy ──────────────────────────────────────────────────────────
    st.subheader("Coordinate Reference System (CRS) Strategy")
    st.markdown(
        """
        Two CRS are used throughout:

        | CRS | EPSG | Use |
        |-----|------|-----|
        | WGS-84 geographic | **4326** | All stored data, GeoPackage outputs, Folium maps |
        | PSAD56 / Peru Central Zone (projected, metres) | **24891** | Metric calculations only — distances, areas |

        **Rule:** data at rest → EPSG:4326; any operation requiring metres → temporary reproject to EPSG:24891.
        """
    )

    # ── Geospatial methodology ────────────────────────────────────────────────
    st.subheader("Geospatial Integration (Task 2)")
    st.markdown(
        """
        1. **Facility → District**: `gpd.sjoin(predicate="within")` + `sjoin_nearest` fallback for border cases.
        2. **Populated Center → District**: same two-step procedure applied to all 136,543 CCPP points
           (221 required the nearest-polygon fallback).
        3. **Nearest emergency facility distance**: `gpd.sjoin_nearest` in EPSG:24891 → distance in metres.
        """
    )

    # ── HADI methodology ─────────────────────────────────────────────────────
    st.subheader("HADI — Healthcare Access Deprivation Index (Task 3)")
    st.markdown(
        r"""
        HADI is a composite index; higher score = more deprived.

        **Three components** (each percentile-ranked to \[0, 1\]):

        | Component | Variable | Direction |
        |-----------|----------|-----------|
        | **C1 — Facility Density** | Emergency facilities per 100 populated centers | Higher → less deprived (rank descending) |
        | **C2 — Emergency Activity** | SUSALUD emergency visits per populated center | Higher → less deprived (rank descending) |
        | **C3 — Spatial Access** | % populated centers > 20 km from nearest facility | Higher → more deprived (rank ascending) |

        **Formula:**
        $$\text{HADI} = \frac{C_1 + C_2 + C_3}{3}$$

        **Quintile classification:** equal-interval bins [0, 0.2, 0.4, 0.6, 0.8, 1.0].
        Q1 = best served; Q5 = most deprived.

        Percentile rank normalization was chosen over min-max scaling because the activity
        distribution is extremely right-skewed (median = 0, max = 262 k visits) — min-max
        would compress 99% of districts into the bottom 1% of the scale.
        """
    )

    # ── How to run the pipeline ───────────────────────────────────────────────
    st.subheader("How to Run")
    st.code(
        """\
# 1 — Create environment (first time only)
conda create -n homework2 python=3.11
conda install -n homework2 -c conda-forge \\
    geopandas folium matplotlib seaborn pandas numpy scipy \\
    pillow pyproj shapely fiona pyogrio openpyxl pyarrow chardet
pip install streamlit streamlit-folium plotly requests branca

# 2 — Run the data pipeline (in order)
conda run -n homework2 python -m src.cleaning       # Task 1
conda run -n homework2 python -m src.geospatial     # Task 2
conda run -n homework2 python -m src.metrics        # Task 3
conda run -n homework2 python -m src.visualization  # Tasks 4 & 5

# 3 — Launch the Streamlit app
conda run -n homework2 streamlit run app.py""",
        language="bash",
    )

    # ── Limitations ──────────────────────────────────────────────────────────
    st.subheader("Limitations")
    st.markdown(
        """
        - **Reporting gap**: SUSALUD data only covers facilities that voluntarily reported;
          the 834 districts with zero confirmed facilities may still have informal emergency services.
        - **Static snapshot**: metrics use the latest available SUSALUD year per district,
          which varies from 2018 to 2026.
        - **Euclidean distance**: straight-line distances from CCPP to facility do not account
          for road network, terrain, or river-crossing constraints — especially relevant in Loreto.
        - **CCPP as population proxy**: populated center counts weight all settlements equally,
          regardless of population size.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Static Analysis")
    st.caption("Six figures answering the four research questions. Each figure is accompanied by its design rationale.")

    # Helper: chart section
    fig_specs = [
        {
            "file": "fig01_supply_distribution.png",
            "title": "Fig 1 — Emergency Facility Supply Distribution",
            "question": "Q1: Facility availability across districts",
            "rationale": (
                "Histogram of SUSALUD-confirmed emergency facilities per district (log y-scale). "
                "The distributional *shape* is the finding: an extreme right skew with a dominant spike at zero "
                "(834 districts, 44.5%, have no confirmed emergency facility). "
                "Log scale keeps both the mass at zero and the long tail simultaneously visible. "
                "A ranking bar chart of individual districts would have buried the structural story."
            ),
        },
        {
            "file": "fig02_supply_vs_activity.png",
            "title": "Fig 2 — Facility Count vs Emergency Care Activity",
            "question": "Q1: Joint supply–activity relationship",
            "rationale": (
                "Scatter of log(n_emergency_active + 1) vs log(SUSALUD visits + 1), "
                "coloured by HADI quintile with an OLS trend line. "
                "The wide vertical scatter confirms a positive association (slope ≈ 2.6) but reveals that "
                "many districts with few facilities still record high activity — and vice versa — "
                "exposing SUSALUD reporting gaps. Two separate histograms would lose the joint relationship."
            ),
        },
        {
            "file": "fig03_spatial_access.png",
            "title": "Fig 3 — Spatial Access: National Distribution + Geographic Pattern",
            "question": "Q2: Which districts have weaker spatial access?",
            "rationale": (
                "Two panels: (A) histogram of % populated centres > 20 km from nearest facility; "
                "(B) horizontal box plots of median CCPP distance by department, sorted by median. "
                "Panel A shows the zero-inflated national distribution (1,188 perfectly covered districts; "
                "125 fully isolated). Panel B adds the geographic dimension, revealing Loreto, "
                "Madre de Dios, and Ucayali as the Amazonian cluster driving inequality. "
                "1,873 individual district bars would be unreadable."
            ),
        },
        {
            "file": "fig04_hadi_distribution.png",
            "title": "Fig 4 — HADI Score Distribution: Baseline vs Alternative",
            "question": "Q3: Deprivation spectrum · Q4: Robustness",
            "rationale": (
                "Histogram of baseline HADI with quintile band shading; overlaid KDE for baseline (solid) "
                "and alternative (dashed). The spike near 0.6 corresponds to districts that have zero "
                "facilities and zero SUSALUD activity but whose populated centres happen to lie within "
                "20 km of a neighbouring facility. The alternative KDE shifts rightward above 0.6, "
                "confirming that the stricter definition reclassifies a subset as more deprived."
            ),
        },
        {
            "file": "fig05_components_by_quintile.png",
            "title": "Fig 5 — HADI Components by Quintile",
            "question": "Q3: What drives the deprivation classification?",
            "rationale": (
                "Grouped bar chart: mean component score (Facility Density, Emergency Activity, Spatial Access) "
                "for each HADI quintile. The key finding — spatial access is *lower* in Q3 than the facility "
                "and activity components, but overtakes them in Q4–Q5 — is immediately visible. "
                "A radar chart would distort relative magnitudes; a heatmap requires colour interpretation "
                "for numeric comparison."
            ),
        },
        {
            "file": "fig06_sensitivity.png",
            "title": "Fig 6 — Sensitivity: Baseline vs Alternative HADI",
            "question": "Q4: Where does the facility definition change results?",
            "rationale": (
                "Scatter of HADI baseline (x) vs HADI alternative (y), coloured by quintile shift (−3 to +3), "
                "with a y = x reference diagonal. Points on the diagonal are unchanged; points above it are "
                "reclassified as more deprived under the alternative. Most districts cluster tightly on "
                "the diagonal (848 unchanged), but 555 worsen by 1 quintile — quantifying sensitivity as "
                "modest but non-negligible."
            ),
        },
    ]

    for spec in fig_specs:
        st.markdown(f"### {spec['title']}")
        st.caption(f"**Research question:** {spec['question']}")
        img_path = FIGURES_DIR / spec["file"]
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"Figure not found: {spec['file']} — run `python -m src.visualization`")
        with st.expander("Design rationale"):
            st.markdown(spec["rationale"])
        st.divider()

    # ── Named district answers to Q1–Q4 ──────────────────────────────────────
    st.subheader("Direct Answers to the Four Research Questions")
    _df = load_hadi()
    label_col = "district_label"

    st.markdown("#### Q1 — Which districts have the highest / lowest facility availability?")
    col_q1a, col_q1b = st.columns(2)
    with col_q1a:
        st.markdown("**Top 10 — most emergency facilities**")
        top_fac = (
            _df.nlargest(10, "n_emergency_active")
            [[label_col, "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={label_col: "District", "n_emergency_active": "Facilities",
                             "susalud_atenciones": "SUSALUD Visits"})
            .reset_index(drop=True)
        )
        top_fac.index += 1
        st.dataframe(top_fac, use_container_width=True)
    with col_q1b:
        st.markdown("**Top 10 — highest emergency care activity (SUSALUD visits)**")
        top_act = (
            _df.nlargest(10, "susalud_atenciones")
            [[label_col, "susalud_atenciones", "n_emergency_active"]]
            .rename(columns={label_col: "District", "susalud_atenciones": "SUSALUD Visits",
                             "n_emergency_active": "Facilities"})
            .reset_index(drop=True)
        )
        top_act.index += 1
        st.dataframe(top_act, use_container_width=True)
    st.caption(f"**834 districts (44.5%) have zero SUSALUD-confirmed emergency facilities.**")

    st.divider()
    st.markdown("#### Q2 — Which districts have populated centers farthest from emergency services?")
    top_isolated = (
        _df.nlargest(15, "pct_ccpp_gt20km_baseline")
        [[label_col, "pct_ccpp_gt20km_baseline", "dist_median_m_baseline", "n_emergency_active"]]
        .rename(columns={
            label_col: "District",
            "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
            "dist_median_m_baseline": "Median Distance (m)",
            "n_emergency_active": "Emergency Facilities",
        })
        .reset_index(drop=True)
    )
    top_isolated.index += 1
    st.dataframe(
        top_isolated, use_container_width=True,
        column_config={
            "% CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
            "Median Distance (m)": st.column_config.NumberColumn(format="%.0f m"),
        },
    )

    st.divider()
    st.markdown("#### Q3 — Which districts are most and least underserved overall (HADI)?")
    col_q3a, col_q3b = st.columns(2)
    with col_q3a:
        st.markdown("**Most deprived — Q5 (top 15 by HADI score)**")
        worst = (
            _df[_df["hadi_quintile_baseline"] == "Q5 (Worst)"]
            .nlargest(15, "hadi_baseline")
            [[label_col, "hadi_baseline", "n_emergency_active", "pct_ccpp_gt20km_baseline"]]
            .rename(columns={label_col: "District", "hadi_baseline": "HADI",
                             "n_emergency_active": "Facilities",
                             "pct_ccpp_gt20km_baseline": "% CCPP > 20 km"})
            .reset_index(drop=True)
        )
        worst.index += 1
        st.dataframe(worst, use_container_width=True,
                     column_config={"HADI": st.column_config.NumberColumn(format="%.3f"),
                                    "% CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%")})
    with col_q3b:
        st.markdown("**Best served — Q1 (top 15 by lowest HADI score)**")
        best = (
            _df[_df["hadi_quintile_baseline"] == "Q1 (Best)"]
            .nsmallest(15, "hadi_baseline")
            [[label_col, "hadi_baseline", "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={label_col: "District", "hadi_baseline": "HADI",
                             "n_emergency_active": "Facilities",
                             "susalud_atenciones": "SUSALUD Visits"})
            .reset_index(drop=True)
        )
        best.index += 1
        st.dataframe(best, use_container_width=True,
                     column_config={"HADI": st.column_config.NumberColumn(format="%.3f")})

    st.divider()
    st.markdown("#### Q4 — Which districts changed most between baseline and alternative?")
    col_q4a, col_q4b = st.columns(2)
    with col_q4a:
        st.markdown("**Most worsened under alternative** (shift ≥ +2)")
        worsened = (
            _df[_df["quintile_shift"] >= 2]
            .sort_values("quintile_shift", ascending=False)
            [[label_col, "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={label_col: "District", "quintile_shift": "Shift",
                             "hadi_baseline": "HADI (Base)", "hadi_alternative": "HADI (Alt)",
                             "hadi_quintile_baseline": "Quintile (Base)",
                             "hadi_quintile_alternative": "Quintile (Alt)"})
            .reset_index(drop=True)
        )
        worsened.index += 1
        st.dataframe(worsened, use_container_width=True,
                     column_config={"HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                                    "HADI (Alt)": st.column_config.NumberColumn(format="%.3f")})
    with col_q4b:
        st.markdown("**Most improved under alternative** (shift ≤ −2)")
        improved = (
            _df[_df["quintile_shift"] <= -2]
            .sort_values("quintile_shift")
            [[label_col, "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={label_col: "District", "quintile_shift": "Shift",
                             "hadi_baseline": "HADI (Base)", "hadi_alternative": "HADI (Alt)",
                             "hadi_quintile_baseline": "Quintile (Base)",
                             "hadi_quintile_alternative": "Quintile (Alt)"})
            .reset_index(drop=True)
        )
        improved.index += 1
        st.dataframe(improved, use_container_width=True,
                     column_config={"HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                                    "HADI (Alt)": st.column_config.NumberColumn(format="%.3f")})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GeoSpatial Results
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("GeoSpatial Results")
    st.caption("Three static maps produced with GeoPandas/matplotlib (Tasks 2 & 5). Interactive maps are in the next tab.")

    map_specs = [
        {
            "file": "map01_hadi_choropleth.png",
            "title": "Map 1 — HADI Quintile Choropleth (Baseline)",
            "question": "Q3: Which districts are most underserved overall?",
            "rationale": (
                "Full-Peru choropleth of baseline HADI quintile (Q1 = blue, Q5 = red) with department "
                "boundary overlay. The map reveals a clear coastal–Amazonian divide: Q1 districts "
                "(best served) cluster along the Pacific coast and in Lima; Q5 districts concentrate "
                "in Loreto, Ucayali, and Amazonas — a geographic pattern invisible in statistical charts."
            ),
        },
        {
            "file": "map02_baseline_vs_alternative.png",
            "title": "Map 2 — Baseline vs Alternative Side-by-Side",
            "question": "Q4: Where does the facility definition change the classification?",
            "rationale": (
                "Two panels at identical scale: left = baseline (3,093 SUSALUD-confirmed), "
                "right = alternative (1,854 structural). Same colour scale enables direct comparison. "
                "The alternative panel shows notable darkening of the Amazon interior and some "
                "lightening of the highlands — the alternative definition captures facilities not "
                "in SUSALUD but structurally equipped."
            ),
        },
        {
            "file": "map03_access_gap.png",
            "title": "Map 3 — Spatial Access Gap",
            "question": "Q2: Which districts have populated centers far from emergency care?",
            "rationale": (
                "Continuous choropleth of % populated centres > 20 km from the nearest emergency "
                "facility. White/green = well-covered; deep red = fully isolated. "
                "Several moderate-HADI districts (Q3) show deep red — they have facilities nearby "
                "in aggregate but their populated centres are scattered far from them. "
                "This decoupling is only visible by mapping the spatial access component separately."
            ),
        },
    ]

    for spec in map_specs:
        st.markdown(f"### {spec['title']}")
        st.caption(f"**Research question:** {spec['question']}")
        img_path = FIGURES_DIR / spec["file"]
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"Map not found: {spec['file']} — run `python -m src.visualization`")
        with st.expander("Design rationale"):
            st.markdown(spec["rationale"])
        st.divider()

    # ── Supporting summary table ───────────────────────────────────────────────
    st.subheader("District Summary by Department")
    df_tab3 = load_hadi()
    dept_summary = (
        df_tab3.groupby("dept_name").agg(
            districts=("ubigeo", "count"),
            zero_facility_districts=("n_emergency_active", lambda x: (x == 0).sum()),
            median_dist_km=("dist_median_m_baseline", lambda x: pd.to_numeric(x, errors="coerce").dropna().div(1000).median()),
            pct_ccpp_gt20km=("pct_ccpp_gt20km_baseline", lambda x: pd.to_numeric(x, errors="coerce").median()),
            q5_districts=("hadi_quintile_baseline", lambda x: (x == "Q5 (Worst)").sum()),
        )
        .reset_index()
        .sort_values("median_dist_km", ascending=False)
        .rename(columns={
            "dept_name": "Department",
            "districts": "Districts",
            "zero_facility_districts": "No Emergency Facility",
            "median_dist_km": "Median Distance (km)",
            "pct_ccpp_gt20km": "Median % CCPP > 20 km",
            "q5_districts": "Q5 Districts",
        })
    )
    dept_summary["Median Distance (km)"] = dept_summary["Median Distance (km)"].round(1)
    dept_summary["Median % CCPP > 20 km"] = dept_summary["Median % CCPP > 20 km"].round(1)
    st.dataframe(
        dept_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Median Distance (km)": st.column_config.NumberColumn(format="%.1f km"),
            "Median % CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Exploration
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Interactive Exploration")

    df = load_hadi()

    # ── Section 1: Interactive Folium maps ─────────────────────────────────────
    st.subheader("Interactive Maps")

    map_choice = st.radio(
        "Select map",
        ["HADI Explorer (Baseline vs Alternative)", "Facility Locations on Access Background"],
        horizontal=True,
    )

    if map_choice == "HADI Explorer (Baseline vs Alternative)":
        st.caption(
            "Hover any district to see UBIGEO, HADI score, quintile, facility count, % CCPP > 20 km, "
            "median distance, and SUSALUD visits. Use the layer control (top-right) to toggle between "
            "Baseline and Alternative HADI."
        )
        html_path = FIGURES_DIR / "map_hadi_explorer.html"
        html_content = load_folium_html(html_path)
        components.html(html_content, height=650, scrolling=False)
    else:
        st.caption(
            "Background: % CCPP > 20 km (green = well-covered, red = isolated). "
            "Blue markers = 3,093 SUSALUD-confirmed facilities. "
            "Orange markers (toggle) = structural-only facilities not in SUSALUD. "
            "Hover markers for name, category, and institution type."
        )
        html_path = FIGURES_DIR / "map_facilities_access.html"
        html_content = load_folium_html(html_path)
        components.html(html_content, height=650, scrolling=False)

    st.divider()

    # ── Section 2: District search & compare ──────────────────────────────────
    st.subheader("District Comparison Tool")
    st.caption("Filter districts by department and quintile, then select up to 5 for a side-by-side comparison.")

    col1, col2, col3 = st.columns(3)
    with col1:
        dept_options = sorted(df["dept_name"].unique())
        selected_depts = st.multiselect("Department(s)", dept_options, default=[])
    with col2:
        q_options = QUINTILE_ORDER
        selected_quintiles = st.multiselect("HADI Quintile (Baseline)", q_options, default=[])
    with col3:
        zero_fac_only = st.checkbox("Only districts with no emergency facility", value=False)

    filt = df.copy()
    if selected_depts:
        filt = filt[filt["dept_name"].isin(selected_depts)]
    if selected_quintiles:
        filt = filt[filt["hadi_quintile_baseline"].isin(selected_quintiles)]
    if zero_fac_only:
        filt = filt[filt["n_emergency_active"] == 0]

    st.markdown(f"**{len(filt):,} districts** match filters")

    display_cols = {
        "ubigeo": "UBIGEO",
        "district_label": "District",
        "n_emergency_active": "Emergency Facilities",
        "n_ccpp_baseline": "Populated Centers",
        "dist_median_m_baseline": "Median Distance (m)",
        "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
        "susalud_atenciones": "SUSALUD Visits",
        "hadi_baseline": "HADI",
        "hadi_quintile_baseline": "Quintile",
        "quintile_shift": "Quintile Shift",
    }
    display_df = filt[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "HADI": st.column_config.NumberColumn(format="%.3f"),
            "Median Distance (m)": st.column_config.NumberColumn(format="%.0f m"),
            "% CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
            "Quintile Shift": st.column_config.NumberColumn(
                help="+N = more deprived under alternative; −N = less deprived"
            ),
        },
    )
    st.download_button(
        "⬇ Download filtered table as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="districts_filtered.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Section 3: Baseline vs Alternative scatter ────────────────────────────
    st.subheader("Baseline vs Alternative HADI Scatter")
    st.caption(
        "Each point is one district. Points on the diagonal are unchanged; "
        "points above it are reclassified as more deprived under the alternative definition."
    )

    # Color by quintile_shift
    shift_palette = {
        -3: "#053061", -2: "#2166ac", -1: "#74add1",
         0: "#cccccc",
         1: "#f4a582",  2: "#d6604d",  3: "#67001f",
    }
    df_plot = df.copy()
    df_plot["color"] = df_plot["quintile_shift"].map(shift_palette).fillna("#cccccc")
    df_plot["shift_label"] = df_plot["quintile_shift"].apply(
        lambda x: f"+{x}" if x > 0 else str(x)
    )

    fig_scatter = go.Figure()
    for shift_val in sorted(df_plot["quintile_shift"].dropna().unique()):
        sub = df_plot[df_plot["quintile_shift"] == shift_val]
        color = shift_palette.get(int(shift_val), "#cccccc")
        label = f"+{int(shift_val)}" if shift_val > 0 else str(int(shift_val))
        fig_scatter.add_trace(go.Scatter(
            x=sub["hadi_baseline"],
            y=sub["hadi_alternative"],
            mode="markers",
            marker=dict(color=color, size=4, opacity=0.7),
            name=f"Shift {label} ({len(sub)} districts)",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "UBIGEO: %{customdata[1]}<br>"
                "Baseline HADI: %{x:.3f}<br>"
                "Alternative HADI: %{y:.3f}<br>"
                "Quintile shift: %{customdata[2]}<extra></extra>"
            ),
            customdata=sub[["district_label", "ubigeo", "shift_label"]].values,
        ))

    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black", dash="dash", width=1),
    )
    fig_scatter.update_layout(
        xaxis_title="HADI — Baseline",
        yaxis_title="HADI — Alternative",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        legend=dict(title="Quintile shift", orientation="v"),
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_scatter.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig_scatter.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Summary stats
    shift_counts = df["quintile_shift"].value_counts().sort_index()
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Unchanged (shift = 0)", f"{shift_counts.get(0, 0):,}")
    with col_s2:
        st.metric("Worsened (shift > 0)", f"{df[df['quintile_shift'] > 0].shape[0]:,}")
    with col_s3:
        st.metric("Improved (shift < 0)", f"{df[df['quintile_shift'] < 0].shape[0]:,}")
    with col_s4:
        st.metric("Max shift", f"{int(df['quintile_shift'].max())}")

    st.divider()

    # ── Section 4: HADI Component radar per quintile ───────────────────────────
    st.subheader("HADI Components by Quintile")
    st.caption("Mean component scores for each quintile. Drag the definition selector to compare baseline vs alternative.")

    defn_choice = st.radio("Definition", ["Baseline", "Alternative"], horizontal=True)
    suffix = "baseline" if defn_choice == "Baseline" else "alternative"

    comp_cols = [f"comp_facility_{suffix}", f"comp_activity_{suffix}", f"comp_access_{suffix}"]
    q_col = f"hadi_quintile_{suffix}"

    comp_means = (
        df.groupby(q_col)[comp_cols]
        .mean()
        .reindex(QUINTILE_ORDER)
        .rename(columns={
            f"comp_facility_{suffix}": "Facility Density",
            f"comp_activity_{suffix}": "Emergency Activity",
            f"comp_access_{suffix}": "Spatial Access",
        })
    )

    comp_fig = px.bar(
        comp_means.reset_index().rename(columns={q_col: "Quintile"}),
        x="Quintile",
        y=["Facility Density", "Emergency Activity", "Spatial Access"],
        barmode="group",
        color_discrete_map={
            "Facility Density": "#4e79a7",
            "Emergency Activity": "#e15759",
            "Spatial Access": "#59a14f",
        },
        labels={"value": "Mean Component Score", "variable": "Component"},
    )
    comp_fig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="national avg = 0.5")
    comp_fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Component",
    )
    comp_fig.update_xaxes(
        categoryorder="array",
        categoryarray=QUINTILE_ORDER,
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    st.divider()

    # ── Section 5: Key metrics panel ──────────────────────────────────────────
    st.subheader("Key Findings")
    nat_median_dist_km = df["dist_median_m_baseline"].median() / 1000
    max_dist_km = df["dist_median_m_baseline"].max() / 1000
    n_zero = (df["n_emergency_active"] == 0).sum()
    n_q5 = (df["hadi_quintile_baseline"] == "Q5 (Worst)").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts analyzed", "1,873")
    c2.metric("Districts with no emergency facility", f"{n_zero} ({n_zero/1873*100:.0f}%)")
    c3.metric("National median distance to facility", f"{nat_median_dist_km:.1f} km")
    c4.metric("Districts in Q5 (most deprived)", f"{n_q5}")

    st.markdown(
        """
        - **Loreto, Ucayali, and Madre de Dios** account for the highest spatial isolation:
          median distances often exceed 50 km; worst-case exceeds 340 km.
        - **555 districts** are reclassified as 1 quintile more deprived when using the
          alternative (structural) facility definition rather than the baseline (SUSALUD-confirmed).
        - The national median distance of **~6.5 km** masks enormous heterogeneity:
          coastal Lima districts have near-zero median distances while Amazonian districts
          can exceed 100 km.
        """
    )
