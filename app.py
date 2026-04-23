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

QUINTILE_ORDER  = ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"]
QUINTILE_COLORS = {
    "Q1 (Best)":  "#2c7bb6",
    "Q2":         "#abd9e9",
    "Q3":         "#ffffbf",
    "Q4":         "#fdae61",
    "Q5 (Worst)": "#d7191c",
}

SHIFT_PALETTE = {
    -3: "#053061", -2: "#2166ac", -1: "#74add1",
     0: "#cccccc",
     1: "#f4a582",  2: "#d6604d",  3: "#67001f",
}

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_hadi() -> pd.DataFrame:
    df = pd.read_csv(TABLES_DIR / "district_hadi.csv", dtype={"ubigeo": str})
    # Accent-correct department names from the numeric IDDPTO code
    df["dept_name"] = df["iddpto"].map(DEPT_NAMES).fillna(df["iddpto"].astype(str))
    df["quintile_shift"] = pd.to_numeric(df["quintile_shift"], errors="coerce")
    # Human-readable label: "Distrito, Provincia (Departamento)"
    if "distrito" in df.columns:
        df["district_label"] = (
            df["distrito"].str.title() + ", " +
            df["provincia"].str.title() + " (" +
            df["dept_name"] + ")"
        )
    else:
        df["district_label"] = df["ubigeo"]
    # Distance in km for display
    df["dist_median_km"] = df["dist_median_m_baseline"] / 1000
    return df


@st.cache_data
def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"*(file not found: {path.name})*"


@st.cache_data
def load_folium_html(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "<p>Map not found — run the pipeline.</p>"


# ── App shell ─────────────────────────────────────────────────────────────────

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

    # Navigation guide
    st.info(
        "**How each research question is answered:**  \n"
        "**Q1** (facility availability) → Tab 2: Figs 1 & 2 + named rankings  \n"
        "**Q2** (spatial access) → Tab 2: Fig 3 + named rankings · Tab 3: Map 3  \n"
        "**Q3** (overall deprivation) → Tab 2: Figs 4 & 5 + rankings · Tab 3: Map 1  \n"
        "**Q4** (sensitivity) → Tab 2: Fig 6 + shift tables · Tab 3: Map 2 · Tab 4: scatter"
    )

    st.subheader("Problem Statement")
    st.markdown(
        """
        Emergency healthcare access in Peru is highly unequal. Peru has **1,873 administrative
        districts** spanning coastal deserts, Andean highlands, and Amazonian rainforest.
        This project quantifies **where** emergency services are scarce and **how far** populations
        must travel to reach them, combining four national datasets into a single district-level
        Healthcare Access Deprivation Index (HADI).

        | # | Research question |
        |---|-------------------|
        | Q1 | Which districts have lower or higher availability of facilities and emergency care activity? |
        | Q2 | Which districts have populated centers with weaker spatial access to emergency services? |
        | Q3 | Which districts are most / least underserved when combining all dimensions? |
        | Q4 | How sensitive are the results to the choice of facility definition? |
        """
    )

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
            **Two facility definitions used in parallel (Q4):**

            | Definition | Criteria | n |
            |------------|----------|--:|
            | **Baseline** | Confirmed in SUSALUD emergency records + valid GPS | 3,093 |
            | **Alternative** | Structural category ≥ I-3, regardless of SUSALUD reporting | 1,854 |

            Baseline = observed activity. Alternative = structural capacity.
            """
        )

    st.subheader("Cleaning & Filtering Decisions")
    st.markdown(
        """
        - **IPRESS**: kept only `estado == ACTIVADO`; removed 26 duplicate `codigo_unico`.
        - **SUSALUD**: dropped 237,753 non-reporter rows (`NE_0001/NE_0002`); dropped 502,040 exact grain duplicates.
        - **CCPP**: dropped 44 coordinate-duplicate populated centers.
        - **DISTRITOS**: kept the largest polygon per UBIGEO (0 duplicates removed).
        """
    )
    with st.expander("Show full cleaning report"):
        st.markdown(load_markdown(TABLES_DIR / "cleaning_report.md"))

    st.subheader("Coordinate Reference System Strategy")
    st.markdown(
        """
        | CRS | EPSG | Use |
        |-----|------|-----|
        | WGS-84 geographic | **4326** | All stored data, GeoPackage outputs, Folium maps |
        | PSAD56 / Peru Central Zone | **24891** | Metric calculations only (distances, areas) |

        **Rule:** data at rest → EPSG:4326; any operation requiring metres → reproject to EPSG:24891, compute, convert back.
        """
    )

    st.subheader("Geospatial Integration (Task 2)")
    st.markdown(
        """
        1. **Facility → District**: `gpd.sjoin(predicate="within")` + `sjoin_nearest` fallback for border cases.
        2. **Populated Center → District**: same two-step procedure for all 136,543 CCPP points (221 used nearest fallback).
        3. **Nearest emergency facility distance**: `gpd.sjoin_nearest` reprojected to EPSG:24891 → distance in metres.
        """
    )

    st.subheader("HADI — Healthcare Access Deprivation Index (Task 3)")
    st.markdown(
        r"""
        HADI ∈ [0, 1]; **higher = more deprived**.

        | Component | Variable | Direction |
        |-----------|----------|-----------|
        | **C1 — Facility Density** | Emergency facilities per 100 populated centers | ↑ facilities → ↓ deprivation |
        | **C2 — Emergency Activity** | SUSALUD visits per populated center | ↑ visits → ↓ deprivation |
        | **C3 — Spatial Access** | % populated centers > 20 km from nearest facility | ↑ isolation → ↑ deprivation |

        Each component is **percentile-ranked** to [0, 1]. Composite:

        $$\text{HADI} = \frac{C_1 + C_2 + C_3}{3}$$

        **Quintiles**: equal-interval bins [0, 0.2, 0.4, 0.6, 0.8, 1.0]. Q1 = best served; Q5 = most deprived.

        *Percentile rank was chosen over min-max scaling because emergency activity is extremely right-skewed
        (median = 0, max = 262,245 visits) — min-max would compress 99% of districts into a tiny band.*
        """
    )

    st.subheader("How to Run")
    st.code(
        """\
# 1 — Create environment
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

# 3 — Launch the app
conda run -n homework2 streamlit run app.py""",
        language="bash",
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - **Reporting gap**: SUSALUD only covers facilities that voluntarily submitted data.
          834 districts with zero confirmed facilities may still have informal emergency services.
        - **Static snapshot**: metrics use the latest available SUSALUD year per district (2018–2026).
        - **Euclidean distance**: straight-line distances ignore roads, rivers, and terrain — especially
          relevant in Loreto where river access is the only option.
        - **CCPP as population proxy**: populated centers are counted equally regardless of size.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Static Analysis")
    _df = load_hadi()

    # ── Fig 1 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 1 — Emergency Facility Supply Distribution")
    st.caption("**Q1:** Facility availability across districts")
    p = FIGURES_DIR / "fig01_supply_distribution.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** 834 districts (44.5%) have **zero** SUSALUD-confirmed emergency facilities. "
        "The distribution is extremely right-skewed — a handful of urban districts concentrate most capacity."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A log-scale histogram shows the distributional shape — the dominant zero-count spike and the long tail — "
            "simultaneously. A bar chart of top districts would miss the structural inequality story."
        )
    st.divider()

    # ── Fig 2 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 2 — Facility Count vs Emergency Care Activity")
    st.caption("**Q1:** Joint supply–activity relationship")
    p = FIGURES_DIR / "fig02_supply_vs_activity.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** More facilities correlate with higher activity (OLS slope ≈ 2.6) but the relationship "
        "is weak. Several districts with many facilities report **zero** SUSALUD visits — a reporting compliance gap, "
        "not a sign of zero activity."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A scatter is the only chart that simultaneously shows both Q1 dimensions and their relationship. "
            "Log1p transform keeps zero-valued districts visible at the origin."
        )
    st.divider()

    # ── Fig 3 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 3 — Spatial Access: National Distribution + Geographic Pattern")
    st.caption("**Q2:** Which districts have weaker spatial access?")
    p = FIGURES_DIR / "fig03_spatial_access.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** 1,188 districts (63%) have **zero** populated centers beyond 20 km — well served. "
        "But 125 districts (7%) have **all** their populated centers beyond 20 km. "
        "Loreto, Madre de Dios, and Ucayali dominate the worst isolation."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Panel A shows the zero-inflated national shape. Panel B adds the geographic dimension via "
            "department box plots — 1,873 individual bars would be unreadable."
        )
    st.divider()

    # ── Fig 4 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 4 — HADI Score Distribution: Baseline vs Alternative")
    st.caption("**Q3:** Overall deprivation spectrum · **Q4:** Sensitivity to facility definition")
    p = FIGURES_DIR / "fig04_hadi_distribution.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** The alternative KDE shifts rightward above HADI 0.6 — the stricter facility "
        "definition reclassifies 555 districts as one quintile more deprived. "
        "The spike near 0.6 captures districts with no facilities but still within 20 km of a neighbor's facility."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A dual KDE overlay is the most direct way to compare two continuous distributions on the same scale "
            "without hiding their shapes. Separate histograms make peak comparison harder."
        )
    st.divider()

    # ── Fig 5 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 5 — HADI Components by Quintile")
    st.caption("**Q3:** What drives the deprivation classification?")
    p = FIGURES_DIR / "fig05_components_by_quintile.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** In Q3 districts, spatial access is *lower* than facility/activity components — "
        "moderate-HADI districts can still be physically isolated. In Q4–Q5, all three components "
        "converge: no facilities, no activity, and far from any service."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Grouped bars allow direct numeric comparison across three components and five quintiles. "
            "A radar chart distorts magnitudes; a heatmap requires colour interpretation."
        )
    st.divider()

    # ── Fig 6 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 6 — Sensitivity: Baseline vs Alternative HADI")
    st.caption("**Q4:** Where does the facility definition change district classifications?")
    p = FIGURES_DIR / "fig06_sensitivity.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Key finding:** 848 districts (45%) are unchanged. 555 (30%) worsen by 1 quintile. "
        "Only 5 districts shift by 3 quintiles — the most extreme reclassifications. "
        "The alternative definition is most impactful for districts near the Q2/Q3 boundary."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Scatter with y = x diagonal: points on the line = unchanged; above = more deprived under alternative. "
            "Colour encodes magnitude of shift. A confusion matrix alone would lose the continuous HADI information."
        )
    st.divider()

    # ── Direct named answers ──────────────────────────────────────────────────
    st.subheader("Direct Answers by District Name")

    # Q1
    st.markdown("#### Q1 — Which districts have the highest / lowest facility availability?")
    col_q1a, col_q1b = st.columns(2)
    with col_q1a:
        st.markdown("**Top 10 — most emergency facilities (baseline)**")
        t = (
            _df.nlargest(10, "n_emergency_active")
            [["district_label", "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={"district_label": "District",
                             "n_emergency_active": "Facilities",
                             "susalud_atenciones": "SUSALUD Visits"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(t, use_container_width=True)
    with col_q1b:
        st.markdown("**Top 10 — highest emergency care activity (SUSALUD visits)**")
        t = (
            _df.nlargest(10, "susalud_atenciones")
            [["district_label", "susalud_atenciones", "n_emergency_active"]]
            .rename(columns={"district_label": "District",
                             "susalud_atenciones": "SUSALUD Visits",
                             "n_emergency_active": "Facilities"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(t, use_container_width=True)

    st.markdown("**Sample of zero-facility districts — 834 total (44.5%)**")
    st.caption("These districts have no SUSALUD-confirmed emergency facility within their boundaries.")
    zero_sample = (
        _df[_df["n_emergency_active"] == 0]
        .nlargest(15, "pct_ccpp_gt20km_baseline")
        [["district_label", "dept_name", "n_ccpp_baseline", "pct_ccpp_gt20km_baseline", "dist_median_km"]]
        .rename(columns={"district_label": "District", "dept_name": "Department",
                         "n_ccpp_baseline": "Populated Centers",
                         "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
                         "dist_median_km": "Median Dist (km)"})
        .reset_index(drop=True)
    )
    zero_sample.index += 1
    st.dataframe(
        zero_sample, use_container_width=True,
        column_config={
            "% CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
            "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
        },
    )

    st.divider()

    # Q2
    st.markdown("#### Q2 — Which districts have populated centers farthest from emergency services?")
    col_q2a, col_q2b = st.columns(2)
    with col_q2a:
        st.markdown("**Most isolated — 100% of CCPP beyond 20 km (top 15 by median distance)**")
        t = (
            _df[_df["pct_ccpp_gt20km_baseline"] == 100]
            .nlargest(15, "dist_median_km")
            [["district_label", "dist_median_km", "n_emergency_active"]]
            .rename(columns={"district_label": "District",
                             "dist_median_km": "Median Dist (km)",
                             "n_emergency_active": "Facilities"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={"Median Dist (km)": st.column_config.NumberColumn(format="%.1f km")},
        )
    with col_q2b:
        st.markdown("**Best accessed — 0% of CCPP beyond 20 km (sample by dept)**")
        t = (
            _df[_df["pct_ccpp_gt20km_baseline"] == 0]
            .nsmallest(15, "dist_median_km")
            [["district_label", "dist_median_km", "n_emergency_active"]]
            .rename(columns={"district_label": "District",
                             "dist_median_km": "Median Dist (km)",
                             "n_emergency_active": "Facilities"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={"Median Dist (km)": st.column_config.NumberColumn(format="%.1f km")},
        )

    st.divider()

    # Q3
    st.markdown("#### Q3 — Which districts are most and least underserved overall?")
    col_q3a, col_q3b = st.columns(2)
    with col_q3a:
        st.markdown("**Most deprived — Q5 (top 15 by HADI score)**")
        t = (
            _df[_df["hadi_quintile_baseline"] == "Q5 (Worst)"]
            .nlargest(15, "hadi_baseline")
            [["district_label", "hadi_baseline", "n_emergency_active",
              "pct_ccpp_gt20km_baseline", "dist_median_km"]]
            .rename(columns={"district_label": "District", "hadi_baseline": "HADI",
                             "n_emergency_active": "Facilities",
                             "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
                             "dist_median_km": "Median Dist (km)"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI": st.column_config.NumberColumn(format="%.3f"),
                "% CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
                "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
            },
        )
    with col_q3b:
        st.markdown("**Best served — Q1 (top 15 by lowest HADI score)**")
        t = (
            _df[_df["hadi_quintile_baseline"] == "Q1 (Best)"]
            .nsmallest(15, "hadi_baseline")
            [["district_label", "hadi_baseline", "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={"district_label": "District", "hadi_baseline": "HADI",
                             "n_emergency_active": "Facilities",
                             "susalud_atenciones": "SUSALUD Visits"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={"HADI": st.column_config.NumberColumn(format="%.3f")},
        )

    st.divider()

    # Q4
    st.markdown("#### Q4 — Which districts changed most between baseline and alternative?")
    col_q4a, col_q4b = st.columns(2)
    with col_q4a:
        st.markdown("**Most worsened under alternative** (shift ≥ +2)")
        t = (
            _df[_df["quintile_shift"] >= 2]
            .sort_values("quintile_shift", ascending=False)
            [["district_label", "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={"district_label": "District", "quintile_shift": "Shift",
                             "hadi_baseline": "HADI (Base)", "hadi_alternative": "HADI (Alt)",
                             "hadi_quintile_baseline": "Q (Base)",
                             "hadi_quintile_alternative": "Q (Alt)"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                "HADI (Alt)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with col_q4b:
        st.markdown("**Most improved under alternative** (shift ≤ −2)")
        t = (
            _df[_df["quintile_shift"] <= -2]
            .sort_values("quintile_shift")
            [["district_label", "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={"district_label": "District", "quintile_shift": "Shift",
                             "hadi_baseline": "HADI (Base)", "hadi_alternative": "HADI (Alt)",
                             "hadi_quintile_baseline": "Q (Base)",
                             "hadi_quintile_alternative": "Q (Alt)"})
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                "HADI (Alt)": st.column_config.NumberColumn(format="%.3f"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GeoSpatial Results
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("GeoSpatial Results")
    st.caption("Three static maps (GeoPandas/matplotlib) + department-level summaries.")

    # ── Map 1 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 1 — HADI Quintile Choropleth (Baseline)")
    st.caption("**Q3:** Which districts are most underserved overall?")
    p = FIGURES_DIR / "map01_hadi_choropleth.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Pattern:** Q1 districts (best served, blue) cluster along the Pacific coast and in Lima. "
        "Q5 districts (most deprived, red) concentrate in Loreto, Ucayali, and Amazonas."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A choropleth is the only chart type that simultaneously shows WHICH districts are deprived "
            "AND WHERE they are. A dot density map adds noise without revealing the polygon-level structure."
        )
    st.divider()

    # ── Map 2 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 2 — Baseline vs Alternative Side-by-Side")
    st.caption("**Q4:** Where does the facility definition change the classification?")
    p = FIGURES_DIR / "map02_baseline_vs_alternative.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Pattern:** The Amazon interior darkens under the alternative definition (fewer qualifying facilities). "
        "Some highland districts lighten — structural category captures facilities that don't report to SUSALUD."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Side-by-side at identical scale externalises the comparison. "
            "A single 'difference' map would lose the absolute deprivation level."
        )
    st.divider()

    # ── Map 3 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 3 — Spatial Access Gap")
    st.caption("**Q2:** Which districts have populated centers far from emergency care?")
    p = FIGURES_DIR / "map03_access_gap.png"
    st.image(str(p), use_container_width=True) if p.exists() else st.warning("Run `python -m src.visualization`")
    st.success(
        "**Pattern:** Several moderate-HADI districts show deep red — they have facilities in the district "
        "but their populated centers are scattered far from them. Physical isolation ≠ zero facilities."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Continuous graduated scale preserves the full gradient. "
            "Using only the HADI map would conflate access with supply and activity components."
        )
    st.divider()

    # ── Department-level summary table ────────────────────────────────────────
    st.subheader("Department-Level Summary")
    df_t3 = load_hadi()
    dept_sum = (
        df_t3.groupby("dept_name").agg(
            Districts=("ubigeo", "count"),
            No_Facility=("n_emergency_active", lambda x: (x == 0).sum()),
            Median_Dist_km=("dist_median_km", lambda x: x.dropna().median()),
            Pct_CCPP_20km=("pct_ccpp_gt20km_baseline", lambda x: pd.to_numeric(x, errors="coerce").median()),
            Q5_Districts=("hadi_quintile_baseline", lambda x: (x == "Q5 (Worst)").sum()),
            Q1_Districts=("hadi_quintile_baseline", lambda x: (x == "Q1 (Best)").sum()),
        )
        .reset_index()
        .sort_values("Median_Dist_km", ascending=False)
        .rename(columns={
            "dept_name": "Department",
            "No_Facility": "No Facility",
            "Median_Dist_km": "Median Dist (km)",
            "Pct_CCPP_20km": "Median % CCPP > 20 km",
            "Q5_Districts": "Q5 (Worst)",
            "Q1_Districts": "Q1 (Best)",
        })
    )
    dept_sum["Median Dist (km)"]      = dept_sum["Median Dist (km)"].round(1)
    dept_sum["Median % CCPP > 20 km"] = dept_sum["Median % CCPP > 20 km"].round(1)

    st.dataframe(
        dept_sum,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Median Dist (km)":      st.column_config.NumberColumn(format="%.1f km"),
            "Median % CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    # ── Q5 by department bar chart ────────────────────────────────────────────
    st.markdown("#### Q5 (Most Deprived) Districts by Department")
    st.caption("Geographic concentration of the worst-served districts.")
    dept_q5 = dept_sum[dept_sum["Q5 (Worst)"] > 0].sort_values("Q5 (Worst)", ascending=True)
    fig_bar = px.bar(
        dept_q5,
        x="Q5 (Worst)", y="Department",
        orientation="h",
        color="Q5 (Worst)",
        color_continuous_scale=[[0, "#fdae61"], [1, "#d7191c"]],
        labels={"Q5 (Worst)": "Q5 Districts"},
        text="Q5 (Worst)",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=420, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=20, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Exploration
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Interactive Exploration")
    df = load_hadi()

    # ── KPI summary at the top ────────────────────────────────────────────────
    n_zero   = int((df["n_emergency_active"] == 0).sum())
    nat_med  = df["dist_median_km"].median()
    n_q5     = int((df["hadi_quintile_baseline"] == "Q5 (Worst)").sum())
    n_change = int((df["quintile_shift"].abs() >= 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts analyzed", "1,873")
    c2.metric("No emergency facility", f"{n_zero:,} (44.5%)")
    c3.metric("National median distance", f"{nat_med:.1f} km")
    c4.metric("Q5 most deprived districts", f"{n_q5:,}")
    st.caption(
        f"**{n_change:,} districts** change quintile between baseline and alternative definitions."
    )
    st.divider()

    # ── Section 1: Interactive Folium maps ────────────────────────────────────
    st.subheader("Interactive Maps")
    map_choice = st.radio(
        "Select map",
        ["HADI Explorer (Baseline vs Alternative)", "Facility Locations on Access Background"],
        horizontal=True,
    )

    if map_choice == "HADI Explorer (Baseline vs Alternative)":
        st.caption(
            "Hover any district to see UBIGEO, HADI score, quintile, facility count, "
            "% CCPP > 20 km, median distance, and SUSALUD visits. "
            "Use the layer control (top-right ≡) to switch between Baseline and Alternative HADI."
        )
        components.html(load_folium_html(FIGURES_DIR / "map_hadi_explorer.html"), height=650, scrolling=False)
    else:
        st.caption(
            "Background choropleth: % CCPP > 20 km (green = well-covered, red = isolated). "
            "Blue markers = 3,093 SUSALUD-confirmed emergency facilities. "
            "Orange markers (toggle via ≡) = structural-only facilities not in SUSALUD. "
            "Hover markers for name, category, and institution type."
        )
        components.html(load_folium_html(FIGURES_DIR / "map_facilities_access.html"), height=650, scrolling=False)

    st.divider()

    # ── Section 2: District search & filter ──────────────────────────────────
    st.subheader("District Explorer")

    col_s, col_d, col_q, col_c = st.columns([2, 2, 2, 1])
    with col_s:
        search_text = st.text_input("Search by district name", placeholder="e.g. Loreto")
    with col_d:
        selected_depts = st.multiselect("Department", sorted(df["dept_name"].unique()))
    with col_q:
        selected_qs = st.multiselect("HADI Quintile (Baseline)", QUINTILE_ORDER)
    with col_c:
        zero_only = st.checkbox("Zero-facility only", value=False)

    filt = df.copy()
    if search_text:
        filt = filt[filt["district_label"].str.contains(search_text, case=False, na=False)]
    if selected_depts:
        filt = filt[filt["dept_name"].isin(selected_depts)]
    if selected_qs:
        filt = filt[filt["hadi_quintile_baseline"].isin(selected_qs)]
    if zero_only:
        filt = filt[filt["n_emergency_active"] == 0]

    st.markdown(f"**{len(filt):,} districts** match · {len(df) - len(filt):,} hidden")

    show_cols = {
        "ubigeo":                   "UBIGEO",
        "district_label":           "District",
        "n_emergency_active":       "Facilities",
        "n_ccpp_baseline":          "CCPP",
        "dist_median_km":           "Median Dist (km)",
        "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
        "susalud_atenciones":       "SUSALUD Visits",
        "hadi_baseline":            "HADI",
        "hadi_quintile_baseline":   "Quintile",
        "quintile_shift":           "Q-Shift",
    }
    disp = filt[list(show_cols)].rename(columns=show_cols).reset_index(drop=True)
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "HADI":            st.column_config.NumberColumn(format="%.3f"),
            "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
            "% CCPP > 20 km":  st.column_config.NumberColumn(format="%.1f%%"),
            "Q-Shift":         st.column_config.NumberColumn(
                help="+N = more deprived under alternative; −N = less deprived"
            ),
        },
    )
    st.download_button(
        "⬇ Download filtered table as CSV",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name="districts_filtered.csv",
        mime="text/csv",
    )

    st.divider()

    # ── Section 3: Baseline vs Alternative scatter ────────────────────────────
    st.subheader("Q4 — Baseline vs Alternative HADI")
    st.caption(
        "Each point is one district. Points on the diagonal (y = x) are unchanged. "
        "Points above the diagonal are reclassified as MORE deprived under the alternative definition."
    )

    df_plot = df.dropna(subset=["hadi_baseline", "hadi_alternative", "quintile_shift"]).copy()
    df_plot["shift_label"] = df_plot["quintile_shift"].apply(
        lambda x: f"+{int(x)}" if x > 0 else str(int(x))
    )

    fig_sc = go.Figure()
    for sv in sorted(df_plot["quintile_shift"].unique()):
        sub   = df_plot[df_plot["quintile_shift"] == sv]
        color = SHIFT_PALETTE.get(int(sv), "#cccccc")
        lbl   = f"+{int(sv)}" if sv > 0 else str(int(sv))
        fig_sc.add_trace(go.Scatter(
            x=sub["hadi_baseline"], y=sub["hadi_alternative"],
            mode="markers",
            marker=dict(color=color, size=5, opacity=0.7),
            name=f"Shift {lbl}  ({len(sub)} districts)",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Baseline HADI: %{x:.3f}<br>"
                "Alternative HADI: %{y:.3f}<br>"
                "Shift: %{customdata[1]}<extra></extra>"
            ),
            customdata=sub[["district_label", "shift_label"]].values,
        ))
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", dash="dash", width=1))
    fig_sc.update_layout(
        xaxis_title="HADI — Baseline", yaxis_title="HADI — Alternative",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        height=500, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Quintile shift"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig_sc.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig_sc.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig_sc, use_container_width=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Unchanged (shift 0)",  f"{(df['quintile_shift'] == 0).sum():,}")
    sc2.metric("Worsened (shift > 0)", f"{(df['quintile_shift']  > 0).sum():,}")
    sc3.metric("Improved (shift < 0)", f"{(df['quintile_shift']  < 0).sum():,}")
    sc4.metric("Max shift",            f"+{int(df['quintile_shift'].max())}")

    st.divider()

    # ── Section 4: HADI components by quintile ───────────────────────────────
    st.subheader("Q3 — HADI Component Breakdown by Quintile")
    st.caption("Mean component scores per quintile. Switch definition to see how components shift.")

    defn = st.radio("Facility definition", ["Baseline", "Alternative"], horizontal=True)
    sfx  = "baseline" if defn == "Baseline" else "alternative"
    q_col = f"hadi_quintile_{sfx}"

    comp_means = (
        df.groupby(q_col)[[f"comp_facility_{sfx}", f"comp_activity_{sfx}", f"comp_access_{sfx}"]]
        .mean()
        .reindex(QUINTILE_ORDER)
        .rename(columns={
            f"comp_facility_{sfx}": "Facility Density",
            f"comp_activity_{sfx}": "Emergency Activity",
            f"comp_access_{sfx}":   "Spatial Access",
        })
    )
    cfig = px.bar(
        comp_means.reset_index().rename(columns={q_col: "Quintile"}),
        x="Quintile",
        y=["Facility Density", "Emergency Activity", "Spatial Access"],
        barmode="group",
        color_discrete_map={
            "Facility Density":  "#4e79a7",
            "Emergency Activity":"#e15759",
            "Spatial Access":    "#59a14f",
        },
        labels={"value": "Mean Component Score", "variable": "Component"},
    )
    cfig.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="national avg = 0.5")
    cfig.update_xaxes(categoryorder="array", categoryarray=QUINTILE_ORDER)
    cfig.update_layout(
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        legend_title_text="Component",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(cfig, use_container_width=True)

    st.divider()

    # ── Section 5: Key findings ───────────────────────────────────────────────
    st.subheader("Key Findings")
    st.markdown(
        f"""
        - **Q1**: {n_zero:,} districts (44.5%) have zero SUSALUD-confirmed emergency facilities.
          Lima, Arequipa, and Callao concentrate most capacity.
        - **Q2**: 125 districts have 100% of populated centers beyond 20 km. Worst case: Purus district
          (Ucayali) with a median distance of **{df.loc[df['distrito']=='Purus', 'dist_median_km'].values[0]:.0f} km** to the nearest facility.
        - **Q3**: Q5 districts are concentrated in Loreto ({int((df[df['hadi_quintile_baseline']=='Q5 (Worst)']['dept_name']=='Loreto').sum())} districts),
          Ucayali, and Amazonas. Best-served are Lima and Callao metropolitan districts.
        - **Q4**: 848 districts unchanged between definitions; 555 worsen by 1 quintile;
          5 shift by 3 quintiles — the alternative definition is most impactful for
          districts near the Q2/Q3 boundary that relied on structural (non-reporting) facilities.
        """
    )
