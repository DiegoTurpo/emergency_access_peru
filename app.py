"""
Emergency Healthcare Access Inequality in Peru
Streamlit application — 4 tabs as required by the assignment.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "output" / "figures"
TABLES_DIR  = ROOT / "output" / "tables"
HADI_CSV    = TABLES_DIR / "district_hadi.csv"

ASSIGNMENT_URL = "https://github.com/d2cml-ai/Data-Science-Python/issues/168"

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
    df = pd.read_csv(HADI_CSV, dtype={"ubigeo": str})

    # Department name — robust to int/str iddpto encodings in the CSV.
    iddpto_int = pd.to_numeric(df["iddpto"], errors="coerce").astype("Int64")
    df["dept_name"] = iddpto_int.map(DEPT_NAMES).fillna(df["iddpto"].astype(str))

    df["quintile_shift"] = pd.to_numeric(df["quintile_shift"], errors="coerce")

    # District label — never render "None". ~42 % of rows have null
    # distrito / provincia (source-data gap, mostly Amazon districts).
    # Fallback: "{Department} · UBIGEO {code}" keeps rows identifiable.
    dist = df.get("distrito", pd.Series(index=df.index, dtype="object"))
    prov = df.get("provincia", pd.Series(index=df.index, dtype="object"))
    has_name = dist.notna() & prov.notna()
    named = (
        dist.fillna("").astype(str).str.title() + ", " +
        prov.fillna("").astype(str).str.title() + " (" + df["dept_name"] + ")"
    )
    fallback = df["dept_name"] + " · UBIGEO " + df["ubigeo"]
    df["district_label"] = np.where(has_name, named, fallback)

    df["dist_median_km"] = df["dist_median_m_baseline"] / 1000
    df["rank_hadi_baseline"] = (
        df["hadi_baseline"].rank(method="min", ascending=True).astype("Int64")
    )
    df["rank_hadi_alternative"] = (
        df["hadi_alternative"].rank(method="min", ascending=True).astype("Int64")
    )
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
    return "<p>Map not found — run the pipeline.</p>"


def show_figure(path: Path):
    """Display a static figure, or a warning if it doesn't exist."""
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(
            "Figure not found — run "
            "`conda run -n homework2 python -m src.visualization`."
        )


@st.cache_data
def compute_kpis(df: pd.DataFrame) -> dict:
    """All dynamic counts/rates used in markdown narratives across tabs."""
    n = len(df)
    shift = df["quintile_shift"]
    return {
        "n_districts":               n,
        "n_zero_facility_baseline":  int((df["n_emergency_active"] == 0).sum()),
        "pct_zero_facility_baseline": float((df["n_emergency_active"] == 0).mean() * 100),
        "n_zero_facility_alt":       int((df["n_emergency_structural"] == 0).sum()),
        "n_ccpp_100pct_isolated":    int((df["pct_ccpp_gt20km_baseline"] == 100).sum()),
        "pct_ccpp_100pct_isolated":  float((df["pct_ccpp_gt20km_baseline"] == 100).mean() * 100),
        "n_ccpp_0pct_isolated":      int((df["pct_ccpp_gt20km_baseline"] == 0).sum()),
        "pct_ccpp_0pct_isolated":    float((df["pct_ccpp_gt20km_baseline"] == 0).mean() * 100),
        "n_baseline_facilities":     int(df["n_emergency_active"].sum()),
        "n_alternative_facilities":  int(df["n_emergency_structural"].sum()),
        "n_shift_unchanged":         int((shift == 0).sum()),
        "pct_shift_unchanged":       float((shift == 0).mean() * 100),
        "n_shift_worse":             int((shift > 0).sum()),
        "n_shift_better":            int((shift < 0).sum()),
        "n_shift_plus1":             int((shift == 1).sum()),
        "n_shift_plus2":             int((shift == 2).sum()),
        "n_shift_plus3":             int((shift == 3).sum()),
        "n_shift_minus1":            int((shift == -1).sum()),
        "n_shift_minus2":            int((shift == -2).sum()),
        "n_shift_minus3":            int((shift == -3).sum()),
        "pct_shift_plus1":           float((shift == 1).mean() * 100),
        "n_q5_baseline":             int((df["hadi_quintile_baseline"] == "Q5 (Worst)").sum()),
        "n_q1_baseline":             int((df["hadi_quintile_baseline"] == "Q1 (Best)").sum()),
        "n_q5_alternative":          int((df["hadi_quintile_alternative"] == "Q5 (Worst)").sum()),
        "max_susalud_visits":        int(df["susalud_atenciones"].max()),
        "nat_median_dist_km":        float(df["dist_median_km"].median()),
        "purus_dist_km":             float(df.loc[df["ubigeo"] == "250401", "dist_median_km"].iloc[0])
                                     if (df["ubigeo"] == "250401").any() else float("nan"),
    }


@st.cache_data
def compute_dept_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-department counts + concentration of Q5."""
    ds = (
        df.groupby("dept_name").agg(
            Districts=("ubigeo", "count"),
            No_Facility=("n_emergency_active", lambda x: int((x == 0).sum())),
            Median_Dist_km=("dist_median_km", lambda x: x.dropna().median()),
            Pct_CCPP_20km=(
                "pct_ccpp_gt20km_baseline",
                lambda x: pd.to_numeric(x, errors="coerce").median(),
            ),
            Q5_Districts=(
                "hadi_quintile_baseline",
                lambda x: int((x == "Q5 (Worst)").sum()),
            ),
            Q1_Districts=(
                "hadi_quintile_baseline",
                lambda x: int((x == "Q1 (Best)").sum()),
            ),
        )
        .reset_index()
        .rename(columns={"dept_name": "Department"})
    )
    ds["Pct_Q5"] = (ds["Q5_Districts"] / ds["Districts"] * 100).round(1)
    ds["Median_Dist_km"] = ds["Median_Dist_km"].round(1)
    ds["Pct_CCPP_20km"]  = ds["Pct_CCPP_20km"].round(1)
    ds = ds.sort_values("Median_Dist_km", ascending=False).reset_index(drop=True)
    return ds


@st.cache_data
def compute_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """5×5 table: rows = baseline quintile, cols = alternative quintile."""
    cm = (
        pd.crosstab(
            df["hadi_quintile_baseline"],
            df["hadi_quintile_alternative"],
            dropna=False,
        )
        .reindex(index=QUINTILE_ORDER, columns=QUINTILE_ORDER, fill_value=0)
        .astype(int)
    )
    return cm


def pipeline_freshness() -> str:
    """Human-readable modification time for the HADI CSV."""
    if not HADI_CSV.exists():
        return "pipeline output not found"
    ts = datetime.fromtimestamp(HADI_CSV.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")


# ── App shell ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emergency Healthcare Access — Peru",
    page_icon="🏥",
    layout="wide",
)

# Short, dense CSS to tighten spacing and improve metric cards.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
      [data-testid="stMetricValue"] { font-size: 1.5rem; }
      [data-testid="stMetricDelta"] { font-size: 0.85rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Emergency Healthcare Access Inequality in Peru")
st.caption(
    "Geospatial analysis across 1,873 districts · IPRESS facility registry · "
    "SUSALUD emergency records 2015–2026 · INEI populated centers"
)

# Load data once — shared across tabs.
df   = load_hadi()
kpis = compute_kpis(df)

# ── Sidebar — global controls ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔧 Global Controls")
    st.caption("Applies across tabs where relevant.")

    definition = st.radio(
        "Facility definition",
        ["Baseline", "Alternative"],
        help=(
            "Baseline = facilities confirmed in SUSALUD emergency records. "
            "Alternative = all structural facilities with category ≥ I-3."
        ),
    )
    sfx = "baseline" if definition == "Baseline" else "alternative"

    dept_filter = st.multiselect(
        "Filter by department (optional)",
        sorted(df["dept_name"].unique()),
        help="Restricts ranking tables and explorer views. Leave empty for national view.",
    )

    st.divider()
    st.markdown("### 📊 Dataset Snapshot")
    st.metric("Districts analyzed", f"{kpis['n_districts']:,}")
    st.metric(
        f"{'SUSALUD' if sfx == 'baseline' else 'Structural ≥ I-3'} facilities",
        f"{kpis['n_baseline_facilities' if sfx == 'baseline' else 'n_alternative_facilities']:,}",
    )
    st.metric(
        "Zero-facility districts",
        f"{kpis['n_zero_facility_baseline' if sfx == 'baseline' else 'n_zero_facility_alt']:,}",
    )

    st.divider()
    st.caption(
        f"🕒 Pipeline refreshed: **{pipeline_freshness()}**  \n"
        f"📂 Source: `output/tables/district_hadi.csv`"
    )
    st.link_button("📋 Open assignment spec", ASSIGNMENT_URL, use_container_width=True)

# Apply dept filter (df_f = filtered view; df remains full national).
df_f = df if not dept_filter else df[df["dept_name"].isin(dept_filter)]
dept_summary = compute_dept_summary(df)  # always national summary

# ── Tabs ──────────────────────────────────────────────────────────────────────

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

    st.info(
        "**How each research question is answered:**  \n"
        "**Q1** (facility availability) → Tab 2: Figs 1 & 2 + ranking tables  \n"
        "**Q2** (spatial access) → Tab 2: Fig 3 · Tab 3: Map 3  \n"
        "**Q3** (overall deprivation) → Tab 2: Figs 4 & 5 · Tab 3: Map 1 + drill-down  \n"
        "**Q4** (sensitivity) → Tab 2: Fig 6 · Tab 3: Map 2 · Tab 4: confusion matrix + scatter"
    )

    st.subheader("Problem Statement")
    st.markdown(
        f"""
        Emergency healthcare access in Peru is highly unequal. Peru has
        **{kpis['n_districts']:,} administrative districts** spanning coastal deserts,
        Andean highlands, and Amazonian rainforest. This project quantifies **where**
        emergency services are scarce and **how far** populations must travel to reach
        them, combining four national datasets into a single district-level Healthcare
        Access Deprivation Index (HADI).

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
            f"""
            | Dataset | Description | Rows |
            |---------|-------------|-----:|
            | **IPRESS** | SUSALUD facility registry (all active) | 20,793 |
            | **SUSALUD** | Emergency care production 2015–2026 | 2,193,587 |
            | **CCPP** | INEI populated centers with coordinates | 136,543 |
            | **DISTRITOS** | INEI district polygons (shapefile) | {kpis['n_districts']:,} |
            """
        )
    with col_b:
        st.markdown(
            f"""
            **Two facility definitions used in parallel (Q4):**

            | Definition | Criteria | n |
            |------------|----------|--:|
            | **Baseline** | Confirmed in SUSALUD emergency records + valid GPS | {kpis['n_baseline_facilities']:,} |
            | **Alternative** | Structural category ≥ I-3, regardless of SUSALUD reporting | {kpis['n_alternative_facilities']:,} |

            Baseline = *observed activity*. Alternative = *structural capacity*.
            """
        )

    st.subheader("Cleaning & Filtering Decisions")
    st.markdown(
        """
        - **IPRESS**: kept only `estado == ACTIVADO`; removed 26 duplicate `codigo_unico`.
        - **SUSALUD**: dropped non-reporter rows (`NE_0001`/`NE_0002`); dropped exact grain duplicates.
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

        **Rule:** data at rest → EPSG:4326; any operation requiring metres → reproject to EPSG:24891,
        compute, convert back.
        """
    )

    st.subheader("Geospatial Integration (Task 2)")
    st.markdown(
        """
        1. **Facility → District**: `gpd.sjoin(predicate="within")` + `sjoin_nearest` fallback
           for border cases.
        2. **Populated Center → District**: same two-step procedure for all 136,543 CCPP points.
        3. **Nearest emergency facility distance**: `gpd.sjoin_nearest` reprojected to EPSG:24891
           → distance in metres.
        """
    )

    st.subheader("HADI — Healthcare Access Deprivation Index (Task 3)")
    st.markdown(
        rf"""
        HADI ∈ [0, 1]; **higher = more deprived**.

        | Component | Variable | Direction |
        |-----------|----------|-----------|
        | **C1 — Facility Density** | Emergency facilities per 100 populated centers | ↑ facilities → ↓ deprivation |
        | **C2 — Emergency Activity** | SUSALUD visits per populated center | ↑ visits → ↓ deprivation |
        | **C3 — Spatial Access** | % populated centers > 20 km from nearest facility | ↑ isolation → ↑ deprivation |

        Each component is **percentile-ranked** to [0, 1]. Composite:

        $$\text{{HADI}} = \frac{{C_1 + C_2 + C_3}}{{3}}$$

        **Quintiles**: equal-interval bins [0, 0.2, 0.4, 0.6, 0.8, 1.0]. Q1 = best served; Q5 = most deprived.

        *Percentile rank was chosen over min-max scaling because emergency activity is extremely
        right-skewed (median = 0, max = {kpis['max_susalud_visits']:,} visits) — min-max would compress
        99 % of districts into a tiny band.*
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
        f"""
        - **Reporting gap**: SUSALUD only covers facilities that voluntarily submitted data.
          {kpis['n_zero_facility_baseline']:,} districts with zero confirmed facilities may still
          have informal emergency services.
        - **Static snapshot**: metrics use the latest available SUSALUD year per district (2018–2026).
        - **Euclidean distance**: straight-line distances ignore roads, rivers, and terrain —
          especially relevant in Loreto where river access is the only option.
        - **CCPP as population proxy**: populated centers are counted equally regardless of size.
        """
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Static Analysis")
    if dept_filter:
        st.caption(
            f"🔎 Ranking tables filtered to **{len(dept_filter)}** department(s): "
            f"{', '.join(dept_filter)}"
        )

    # ── Fig 1 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 1 — Emergency Facility Supply Distribution")
    st.caption("**Q1:** Facility availability across districts")
    show_figure(FIGURES_DIR / "fig01_supply_distribution.png")
    st.success(
        f"**Key finding:** {kpis['n_zero_facility_baseline']:,} districts "
        f"({kpis['pct_zero_facility_baseline']:.1f} %) have **zero** SUSALUD-confirmed "
        f"emergency facilities. The distribution is extremely right-skewed — a handful of "
        f"urban districts concentrate most of the {kpis['n_baseline_facilities']:,} facilities."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A log-scale histogram shows the distributional shape — the dominant zero-count "
            "spike and the long tail — simultaneously. A bar chart of top districts would miss "
            "the structural inequality story."
        )
    st.divider()

    # ── Fig 2 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 2 — Facility Count vs Emergency Care Activity")
    st.caption("**Q1:** Joint supply–activity relationship")
    show_figure(FIGURES_DIR / "fig02_supply_vs_activity.png")
    st.success(
        "**Key finding:** More facilities correlate with higher activity (OLS slope ≈ 2.6) "
        "but the relationship is weak. Several districts with many facilities report **zero** "
        "SUSALUD visits — a reporting compliance gap, not a sign of zero activity."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A scatter is the only chart that simultaneously shows both Q1 dimensions and "
            "their relationship. Log1p transform keeps zero-valued districts visible at the origin."
        )
    st.divider()

    # ── Fig 3 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 3 — Spatial Access: National Distribution + Geographic Pattern")
    st.caption("**Q2:** Which districts have weaker spatial access?")
    show_figure(FIGURES_DIR / "fig03_spatial_access.png")
    st.success(
        f"**Key finding:** {kpis['n_ccpp_0pct_isolated']:,} districts "
        f"({kpis['pct_ccpp_0pct_isolated']:.0f} %) have **zero** populated centers beyond 20 km "
        f"— well served. But {kpis['n_ccpp_100pct_isolated']:,} districts "
        f"({kpis['pct_ccpp_100pct_isolated']:.0f} %) have **all** their populated centers beyond "
        f"20 km. Loreto, Madre de Dios, and Ucayali dominate the worst isolation."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Panel A shows the zero-inflated national shape. Panel B adds the geographic "
            "dimension via department box plots — 1,873 individual bars would be unreadable."
        )
    st.divider()

    # ── Fig 4 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 4 — HADI Score Distribution: Baseline vs Alternative")
    st.caption("**Q3:** Overall deprivation spectrum · **Q4:** Sensitivity to facility definition")
    show_figure(FIGURES_DIR / "fig04_hadi_distribution.png")
    st.success(
        f"**Key finding:** The alternative KDE shifts rightward above HADI 0.6 — the stricter "
        f"facility definition reclassifies {kpis['n_shift_plus1']:,} districts as one quintile "
        f"more deprived. The spike near 0.6 captures districts with no facilities but still "
        f"within 20 km of a neighbour's facility."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "A dual KDE overlay is the most direct way to compare two continuous distributions "
            "on the same scale without hiding their shapes."
        )
    st.divider()

    # ── Fig 5 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 5 — HADI Components by Quintile")
    st.caption("**Q3:** What drives the deprivation classification?")
    show_figure(FIGURES_DIR / "fig05_components_by_quintile.png")
    st.success(
        "**Key finding:** In Q3 districts, spatial access is *lower* than facility/activity "
        "components — moderate-HADI districts can still be physically isolated. "
        "In Q4–Q5, all three components converge: no facilities, no activity, and far from "
        "any service."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Grouped bars allow direct numeric comparison across three components and five "
            "quintiles. A radar chart distorts magnitudes; a heatmap requires colour interpretation."
        )
    st.divider()

    # ── Fig 6 ─────────────────────────────────────────────────────────────────
    st.markdown("### Fig 6 — Sensitivity: Baseline vs Alternative HADI")
    st.caption("**Q4:** Where does the facility definition change district classifications?")
    show_figure(FIGURES_DIR / "fig06_sensitivity.png")
    st.success(
        f"**Key finding:** {kpis['n_shift_unchanged']:,} districts "
        f"({kpis['pct_shift_unchanged']:.0f} %) are unchanged. "
        f"{kpis['n_shift_plus1']:,} ({kpis['pct_shift_plus1']:.0f} %) worsen by 1 quintile. "
        f"Only {kpis['n_shift_plus3']} districts shift by 3 quintiles — the most extreme "
        f"reclassifications. The alternative definition is most impactful for districts near "
        f"the Q2/Q3 boundary."
    )
    with st.expander("Design rationale"):
        st.markdown(
            "Scatter with y = x diagonal: points on the line = unchanged; above = more deprived "
            "under alternative. Colour encodes magnitude of shift."
        )
    st.divider()

    # ── NEW: Institution mix by quintile ─────────────────────────────────────
    st.markdown("### Fig 7 — Who Runs the Facilities?  (Public vs Private mix by quintile)")
    st.caption("**Q1 sub-question:** Institutional composition of emergency supply.")
    mix = (
        df.groupby("hadi_quintile_baseline").agg(
            Public =("n_public",  "sum"),
            Private=("n_private", "sum"),
        )
        .reindex(QUINTILE_ORDER)
        .reset_index()
        .rename(columns={"hadi_quintile_baseline": "Quintile"})
    )
    mix_fig = go.Figure()
    mix_fig.add_bar(name="Public",  x=mix["Quintile"], y=mix["Public"],  marker_color="#2c7bb6")
    mix_fig.add_bar(name="Private", x=mix["Quintile"], y=mix["Private"], marker_color="#fdae61")
    mix_fig.update_layout(
        barmode="stack", height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Total Facilities",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(mix_fig, use_container_width=True)
    total_public  = int(df["n_public"].sum())
    total_private = int(df["n_private"].sum())
    st.success(
        f"**Key finding:** National mix is **{total_public:,} public** and "
        f"**{total_private:,} private** facilities. Q5 districts are overwhelmingly public "
        f"(private providers locate where activity is highest — Q1/Q2)."
    )
    st.divider()

    # ── Direct named answers ──────────────────────────────────────────────────
    st.subheader("Direct Answers by District Name")
    df_rank = df_f  # apply sidebar dept filter to all rankings below

    # Q1
    st.markdown("#### Q1 — Which districts have the highest / lowest facility availability?")
    col_q1a, col_q1b = st.columns(2)
    with col_q1a:
        st.markdown("**Top 10 — most emergency facilities**")
        t = (
            df_rank.nlargest(10, "n_emergency_active")
            [["district_label", "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={
                "district_label":      "District",
                "n_emergency_active":  "Facilities",
                "susalud_atenciones":  "SUSALUD Visits",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(t, use_container_width=True)
    with col_q1b:
        st.markdown("**Top 10 — highest emergency care activity (SUSALUD visits)**")
        t = (
            df_rank.nlargest(10, "susalud_atenciones")
            [["district_label", "susalud_atenciones", "n_emergency_active"]]
            .rename(columns={
                "district_label":      "District",
                "susalud_atenciones":  "SUSALUD Visits",
                "n_emergency_active":  "Facilities",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(t, use_container_width=True)

    st.markdown(
        f"**Sample of zero-facility districts — {kpis['n_zero_facility_baseline']:,} total "
        f"({kpis['pct_zero_facility_baseline']:.1f} %)**"
    )
    st.caption("These districts have no SUSALUD-confirmed emergency facility within their boundaries.")
    zero_sample = (
        df_rank[df_rank["n_emergency_active"] == 0]
        .nlargest(15, "pct_ccpp_gt20km_baseline")
        [["district_label", "dept_name", "n_ccpp_baseline", "pct_ccpp_gt20km_baseline", "dist_median_km"]]
        .rename(columns={
            "district_label":           "District",
            "dept_name":                "Department",
            "n_ccpp_baseline":          "Populated Centers",
            "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
            "dist_median_km":           "Median Dist (km)",
        })
        .reset_index(drop=True)
    )
    zero_sample.index += 1
    st.dataframe(
        zero_sample, use_container_width=True,
        column_config={
            "% CCPP > 20 km":   st.column_config.NumberColumn(format="%.1f%%"),
            "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
        },
    )
    st.divider()

    # Q2
    st.markdown("#### Q2 — Which districts have populated centers farthest from emergency services?")
    col_q2a, col_q2b = st.columns(2)
    with col_q2a:
        st.markdown("**Most isolated — 100 % of CCPP beyond 20 km (top 15 by median distance)**")
        t = (
            df_rank[df_rank["pct_ccpp_gt20km_baseline"] == 100]
            .nlargest(15, "dist_median_km")
            [["district_label", "dist_median_km", "n_emergency_active"]]
            .rename(columns={
                "district_label":     "District",
                "dist_median_km":     "Median Dist (km)",
                "n_emergency_active": "Facilities",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={"Median Dist (km)": st.column_config.NumberColumn(format="%.1f km")},
        )
    with col_q2b:
        st.markdown("**Best accessed — 0 % of CCPP beyond 20 km (sample by dist)**")
        t = (
            df_rank[df_rank["pct_ccpp_gt20km_baseline"] == 0]
            .nsmallest(15, "dist_median_km")
            [["district_label", "dist_median_km", "n_emergency_active"]]
            .rename(columns={
                "district_label":     "District",
                "dist_median_km":     "Median Dist (km)",
                "n_emergency_active": "Facilities",
            })
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
            df_rank[df_rank["hadi_quintile_baseline"] == "Q5 (Worst)"]
            .nlargest(15, "hadi_baseline")
            [["district_label", "hadi_baseline", "n_emergency_active",
              "pct_ccpp_gt20km_baseline", "dist_median_km"]]
            .rename(columns={
                "district_label":           "District",
                "hadi_baseline":            "HADI",
                "n_emergency_active":       "Facilities",
                "pct_ccpp_gt20km_baseline": "% CCPP > 20 km",
                "dist_median_km":           "Median Dist (km)",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI":             st.column_config.NumberColumn(format="%.3f"),
                "% CCPP > 20 km":   st.column_config.NumberColumn(format="%.1f%%"),
                "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
            },
        )
    with col_q3b:
        st.markdown("**Best served — Q1 (top 15 by lowest HADI score)**")
        t = (
            df_rank[df_rank["hadi_quintile_baseline"] == "Q1 (Best)"]
            .nsmallest(15, "hadi_baseline")
            [["district_label", "hadi_baseline", "n_emergency_active", "susalud_atenciones"]]
            .rename(columns={
                "district_label":     "District",
                "hadi_baseline":      "HADI",
                "n_emergency_active": "Facilities",
                "susalud_atenciones": "SUSALUD Visits",
            })
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
            df_rank[df_rank["quintile_shift"] >= 2]
            .sort_values("quintile_shift", ascending=False)
            [["district_label", "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={
                "district_label":            "District",
                "quintile_shift":            "Shift",
                "hadi_baseline":             "HADI (Base)",
                "hadi_alternative":          "HADI (Alt)",
                "hadi_quintile_baseline":    "Q (Base)",
                "hadi_quintile_alternative": "Q (Alt)",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                "HADI (Alt)":  st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with col_q4b:
        st.markdown("**Most improved under alternative** (shift ≤ −2)")
        t = (
            df_rank[df_rank["quintile_shift"] <= -2]
            .sort_values("quintile_shift")
            [["district_label", "quintile_shift", "hadi_baseline", "hadi_alternative",
              "hadi_quintile_baseline", "hadi_quintile_alternative"]]
            .rename(columns={
                "district_label":            "District",
                "quintile_shift":            "Shift",
                "hadi_baseline":             "HADI (Base)",
                "hadi_alternative":          "HADI (Alt)",
                "hadi_quintile_baseline":    "Q (Base)",
                "hadi_quintile_alternative": "Q (Alt)",
            })
            .reset_index(drop=True)
        )
        t.index += 1
        st.dataframe(
            t, use_container_width=True,
            column_config={
                "HADI (Base)": st.column_config.NumberColumn(format="%.3f"),
                "HADI (Alt)":  st.column_config.NumberColumn(format="%.3f"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GeoSpatial Results
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("GeoSpatial Results")
    st.caption("Three static maps (GeoPandas / matplotlib) + department summary + district drill-down.")

    # ── Map 1 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 1 — HADI Quintile Choropleth (Baseline)")
    st.caption("**Q3:** Which districts are most underserved overall?")
    show_figure(FIGURES_DIR / "map01_hadi_choropleth.png")
    st.success(
        "**Pattern:** Q1 districts (best served, blue) cluster along the Pacific coast and in "
        "Lima. Q5 districts (most deprived, red) concentrate in Loreto, Ucayali, and Amazonas."
    )
    st.divider()

    # ── Map 2 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 2 — Baseline vs Alternative Side-by-Side")
    st.caption("**Q4:** Where does the facility definition change the classification?")
    show_figure(FIGURES_DIR / "map02_baseline_vs_alternative.png")
    st.success(
        "**Pattern:** The Amazon interior darkens under the alternative definition (fewer "
        "qualifying facilities). Some highland districts lighten — structural category captures "
        "facilities that don't report to SUSALUD."
    )
    st.divider()

    # ── Map 3 ─────────────────────────────────────────────────────────────────
    st.markdown("### Map 3 — Spatial Access Gap")
    st.caption("**Q2:** Which districts have populated centers far from emergency care?")
    show_figure(FIGURES_DIR / "map03_access_gap.png")
    st.success(
        "**Pattern:** Several moderate-HADI districts show deep red — they have facilities in "
        "the district but their populated centers are scattered far from them. Physical "
        "isolation ≠ zero facilities."
    )
    st.divider()

    # ── Department summary ───────────────────────────────────────────────────
    st.subheader("Department-Level Summary")
    dept_view = dept_summary.copy()
    if dept_filter:
        dept_view = dept_view[dept_view["Department"].isin(dept_filter)]

    dept_view_renamed = dept_view.rename(columns={
        "No_Facility":    "No Facility",
        "Median_Dist_km": "Median Dist (km)",
        "Pct_CCPP_20km":  "Median % CCPP > 20 km",
        "Q5_Districts":   "Q5 (Worst)",
        "Q1_Districts":   "Q1 (Best)",
        "Pct_Q5":         "% Q5",
    })
    st.dataframe(
        dept_view_renamed,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Median Dist (km)":      st.column_config.NumberColumn(format="%.1f km"),
            "Median % CCPP > 20 km": st.column_config.NumberColumn(format="%.1f%%"),
            "% Q5":                  st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    # ── Q5 bar charts (count + concentration) ────────────────────────────────
    col_bc1, col_bc2 = st.columns(2)
    with col_bc1:
        st.markdown("#### Q5 (Most Deprived) — Count by Department")
        dept_q5 = dept_summary[dept_summary["Q5_Districts"] > 0].sort_values(
            "Q5_Districts", ascending=True
        )
        fig_bar = px.bar(
            dept_q5, x="Q5_Districts", y="Department", orientation="h",
            color="Q5_Districts",
            color_continuous_scale=[[0, "#fdae61"], [1, "#d7191c"]],
            text="Q5_Districts",
            labels={"Q5_Districts": "Q5 Districts"},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            height=420, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=10, r=40, t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_bc2:
        st.markdown("#### Q5 Concentration — Share of Dept's Districts in Q5")
        dept_pct = dept_summary[dept_summary["Pct_Q5"] > 0].sort_values(
            "Pct_Q5", ascending=True
        )
        fig_pct = px.bar(
            dept_pct, x="Pct_Q5", y="Department", orientation="h",
            color="Pct_Q5",
            color_continuous_scale=[[0, "#fdae61"], [1, "#67001f"]],
            text="Pct_Q5",
            labels={"Pct_Q5": "% of Dept's Districts in Q5"},
        )
        fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_pct.update_layout(
            height=420, showlegend=False, coloraxis_showscale=False,
            margin=dict(l=10, r=40, t=20, b=30),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_pct, use_container_width=True)
    st.caption(
        "**Count** = absolute number of Q5 districts. **Concentration** = share of that "
        "department's districts in Q5 — surfaces small departments that are uniformly underserved."
    )
    st.divider()

    # ── NEW: District drill-down ──────────────────────────────────────────────
    st.subheader("District Drill-Down")
    st.caption(
        f"Pick any of the {kpis['n_districts']:,} districts to see its full HADI profile, "
        "national rank, facility mix, and nearest-HADI neighbours."
    )

    labels_sorted = df.sort_values("district_label")["district_label"].tolist()
    sel_label = st.selectbox("Select a district", labels_sorted, key="drill_down_select")
    row = df.loc[df["district_label"] == sel_label].iloc[0]

    dd_c1, dd_c2, dd_c3 = st.columns(3)
    with dd_c1:
        st.markdown("**HADI Profile**")
        st.metric(
            "HADI (Baseline)",
            f"{row['hadi_baseline']:.3f}",
            delta=f"{row['hadi_alternative'] - row['hadi_baseline']:+.3f} vs Alt",
            delta_color="inverse",
        )
        st.metric("Quintile (Baseline)",   str(row["hadi_quintile_baseline"]))
        st.metric("Quintile (Alternative)", str(row["hadi_quintile_alternative"]))
        shift_val = row["quintile_shift"]
        shift_str = "unchanged" if pd.isna(shift_val) or shift_val == 0 else (
            f"+{int(shift_val)} (more deprived under Alt)" if shift_val > 0 else
            f"{int(shift_val)} (less deprived under Alt)"
        )
        st.caption(f"Sensitivity: **{shift_str}**")

    with dd_c2:
        st.markdown("**District Facts**")
        st.metric("Emergency facilities (Base)", f"{int(row['n_emergency_active'])}")
        st.metric(
            "Structural facilities (Alt)",
            f"{int(row['n_emergency_structural'])}",
            delta=f"{int(row['n_emergency_structural'] - row['n_emergency_active']):+d}",
        )
        st.metric("Populated centers",  f"{int(row['n_ccpp_baseline'])}")
        st.metric("Beds",                f"{int(row['camas_total'])}")
        pub, priv = int(row["n_public"]), int(row["n_private"])
        st.caption(f"Public / Private: **{pub} / {priv}**")

    with dd_c3:
        st.markdown("**National Ranks**")
        st.metric(
            "HADI Baseline rank",
            f"{int(row['rank_hadi_baseline'])} / {kpis['n_districts']:,}",
            help="1 = best served, higher = more deprived",
        )
        st.metric(
            "HADI Alternative rank",
            f"{int(row['rank_hadi_alternative'])} / {kpis['n_districts']:,}",
        )
        median_dist = row["dist_median_km"]
        pct20       = row["pct_ccpp_gt20km_baseline"]
        st.metric(
            "Median distance to facility",
            f"{median_dist:.1f} km" if pd.notna(median_dist) else "—",
        )
        st.metric(
            "% CCPP > 20 km from facility",
            f"{pct20:.1f} %" if pd.notna(pct20) else "—",
        )

    st.markdown("**5 most similar districts (by Baseline HADI score)**")
    target = row["hadi_baseline"]
    if pd.notna(target):
        nearest = (
            df.loc[df["ubigeo"] != row["ubigeo"]]
            .assign(_dist=lambda d: (d["hadi_baseline"] - target).abs())
            .nsmallest(5, "_dist")
            [["district_label", "hadi_baseline", "hadi_quintile_baseline",
              "n_emergency_active", "dist_median_km"]]
            .rename(columns={
                "district_label":           "District",
                "hadi_baseline":            "HADI",
                "hadi_quintile_baseline":   "Quintile",
                "n_emergency_active":       "Facilities",
                "dist_median_km":           "Median Dist (km)",
            })
            .reset_index(drop=True)
        )
        nearest.index += 1
        st.dataframe(
            nearest,
            use_container_width=True,
            column_config={
                "HADI":             st.column_config.NumberColumn(format="%.3f"),
                "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
            },
        )
    else:
        st.info("HADI is not available for this district (insufficient data).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Exploration
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Interactive Exploration")

    # ── KPI summary ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts analyzed", f"{kpis['n_districts']:,}")
    c2.metric(
        "No emergency facility",
        f"{kpis['n_zero_facility_baseline']:,}",
        delta=f"{kpis['pct_zero_facility_baseline']:.1f}%",
        delta_color="off",
    )
    c3.metric("National median distance", f"{kpis['nat_median_dist_km']:.1f} km")
    c4.metric("Q5 most deprived", f"{kpis['n_q5_baseline']:,}")
    st.caption(
        f"**{kpis['n_districts'] - kpis['n_shift_unchanged']:,} districts** "
        f"change quintile between baseline and alternative definitions "
        f"(agreement rate = {kpis['pct_shift_unchanged']:.1f} %)."
    )
    st.divider()

    # ── NEW: Confusion matrix — Q4 summary ────────────────────────────────────
    st.subheader("Q4 — Baseline × Alternative Confusion Matrix")
    st.caption(
        "Rows = baseline quintile · Columns = alternative quintile. "
        "Diagonal cells = districts unchanged between definitions."
    )
    cm = compute_confusion_matrix(df)
    cm_fig = px.imshow(
        cm.values,
        x=[q.replace(" (Best)", "").replace(" (Worst)", "") for q in cm.columns],
        y=[q.replace(" (Best)", "").replace(" (Worst)", "") for q in cm.index],
        labels=dict(x="Alternative Quintile", y="Baseline Quintile", color="Districts"),
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="equal",
    )
    cm_fig.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=40, r=40, t=30, b=40),
        coloraxis_colorbar=dict(title="Districts"),
    )
    cm_fig.update_xaxes(side="top")
    st.plotly_chart(cm_fig, use_container_width=True)

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Unchanged (diagonal)", f"{kpis['n_shift_unchanged']:,}",
               delta=f"{kpis['pct_shift_unchanged']:.1f}%", delta_color="off")
    cc2.metric("Worsened under Alt",   f"{kpis['n_shift_worse']:,}")
    cc3.metric("Improved under Alt",   f"{kpis['n_shift_better']:,}")
    cc4.metric("Extreme shifts (|Δ|=3)",
               f"{kpis['n_shift_plus3'] + kpis['n_shift_minus3']:,}")
    st.divider()

    # ── Interactive Folium maps ───────────────────────────────────────────────
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
        components.html(
            load_folium_html(FIGURES_DIR / "map_hadi_explorer.html"),
            height=650, scrolling=False,
        )
    else:
        st.caption(
            f"Background choropleth: % CCPP > 20 km (green = well-covered, red = isolated). "
            f"Blue markers = {kpis['n_baseline_facilities']:,} SUSALUD-confirmed emergency facilities. "
            f"Orange markers (toggle via ≡) = structural-only facilities not in SUSALUD. "
            f"Hover markers for name, category, and institution type."
        )
        components.html(
            load_folium_html(FIGURES_DIR / "map_facilities_access.html"),
            height=650, scrolling=False,
        )
    st.divider()

    # ── District explorer ─────────────────────────────────────────────────────
    st.subheader("District Explorer")

    col_s, col_q, col_c = st.columns([2, 2, 1])
    with col_s:
        search_text = st.text_input("Search by district name", placeholder="e.g. Purus")
    with col_q:
        selected_qs = st.multiselect(f"HADI Quintile ({definition})", QUINTILE_ORDER)
    with col_c:
        zero_only = st.checkbox("Zero-facility only", value=False)

    q_col = f"hadi_quintile_{sfx}"
    filt = df_f.copy()
    if search_text:
        filt = filt[filt["district_label"].str.contains(search_text, case=False, na=False)]
    if selected_qs:
        filt = filt[filt[q_col].isin(selected_qs)]
    if zero_only:
        filt = filt[filt["n_emergency_active"] == 0]

    st.markdown(f"**{len(filt):,} districts** match · {kpis['n_districts'] - len(filt):,} hidden")

    rank_col = f"rank_hadi_{sfx}"
    show_cols = {
        rank_col:                      "Rank",
        "ubigeo":                      "UBIGEO",
        "district_label":              "District",
        "n_emergency_active":          "Facilities",
        "n_ccpp_baseline":             "CCPP",
        "dist_median_km":              "Median Dist (km)",
        "pct_ccpp_gt20km_baseline":    "% CCPP > 20 km",
        "susalud_atenciones":          "SUSALUD Visits",
        f"hadi_{sfx}":                 "HADI",
        q_col:                         "Quintile",
        "quintile_shift":              "Q-Shift",
    }
    disp = (
        filt[list(show_cols)]
        .rename(columns=show_cols)
        .sort_values("Rank")
        .reset_index(drop=True)
    )
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "HADI":             st.column_config.NumberColumn(format="%.3f"),
            "Median Dist (km)": st.column_config.NumberColumn(format="%.1f km"),
            "% CCPP > 20 km":   st.column_config.NumberColumn(format="%.1f%%"),
            "Q-Shift":          st.column_config.NumberColumn(
                                    help="+N = more deprived under alternative; −N = less deprived"),
        },
    )

    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        st.download_button(
            "⬇ Download filtered table (CSV)",
            data=disp.to_csv(index=False).encode("utf-8"),
            file_name="districts_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_c2:
        st.download_button(
            f"⬇ Download full HADI table ({kpis['n_districts']:,} rows, CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="district_hadi.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.divider()

    # ── Baseline vs Alternative scatter ───────────────────────────────────────
    st.subheader("Q4 Detail — Baseline vs Alternative HADI (scatter)")
    st.caption(
        "Each point is one district. Points on the diagonal (y = x) are unchanged. "
        "Points above the diagonal are reclassified as MORE deprived under the alternative."
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
    st.divider()

    # ── HADI components by quintile (driven by sidebar definition) ────────────
    st.subheader(f"Q3 — HADI Component Breakdown by Quintile ({definition})")
    st.caption("Mean component scores per quintile. Switch definition in the sidebar to re-draw.")

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
            "Facility Density":   "#4e79a7",
            "Emergency Activity": "#e15759",
            "Spatial Access":     "#59a14f",
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

    # ── Key findings (fully dynamic) ─────────────────────────────────────────
    st.subheader("Key Findings (dynamic)")
    top_q5_depts = (
        df[df["hadi_quintile_baseline"] == "Q5 (Worst)"]
        .groupby("dept_name").size().sort_values(ascending=False)
    )
    loreto_q5   = int(top_q5_depts.get("Loreto", 0))
    top3_str    = ", ".join(
        f"{d} ({n})" for d, n in top_q5_depts.head(3).items()
    ) if len(top_q5_depts) else "—"
    purus_dist  = (
        f"{kpis['purus_dist_km']:.0f}" if not np.isnan(kpis["purus_dist_km"]) else "~232"
    )

    st.markdown(
        f"""
        - **Q1** — **{kpis['n_zero_facility_baseline']:,} districts
          ({kpis['pct_zero_facility_baseline']:.1f} %)** have zero SUSALUD-confirmed emergency
          facilities. Lima, Arequipa, and Callao concentrate most capacity.
        - **Q2** — **{kpis['n_ccpp_100pct_isolated']:,} districts** have 100 % of populated
          centers beyond 20 km. Worst case: **Purus** (Ucayali) with a median distance of
          **{purus_dist} km** to the nearest facility.
        - **Q3** — Q5 districts are concentrated in **{top3_str}**. Best-served are Lima and
          Callao metropolitan districts (Q1).
        - **Q4** — **{kpis['n_shift_unchanged']:,} districts unchanged**
          ({kpis['pct_shift_unchanged']:.0f} %); **{kpis['n_shift_plus1']:,} worsen by 1 quintile**;
          **{kpis['n_shift_plus3']} shift by 3 quintiles** — the alternative definition is most
          impactful for districts near the Q2/Q3 boundary that relied on structural
          (non-reporting) facilities.
        """
    )
