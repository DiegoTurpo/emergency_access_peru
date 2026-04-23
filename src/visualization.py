"""
Task 4 — Static visual analysis (matplotlib + seaborn).

Six figures, each tied to one or more of the four required analytical questions:

  fig01_supply_distribution.png    Q1  Facility availability across districts
  fig02_supply_vs_activity.png     Q1  Does facility count predict care delivery?
  fig03_spatial_access.png         Q2  Spatial isolation: national dist. + by dept.
  fig04_hadi_distribution.png      Q3  HADI spectrum + baseline vs alternative
  fig05_components_by_quintile.png Q3  What drives deprivation classification?
  fig06_sensitivity.png            Q4  Baseline vs alternative specification shift
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch as MPatch
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import branca.colormap as bcm
from pathlib import Path

FIGURES_DIR   = Path(__file__).resolve().parents[1] / "output" / "figures"
DATA_DIR      = Path(__file__).resolve().parents[1] / "output" / "tables"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

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

QUINTILE_ORDER = ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"]

# RdYlBu diverging — blue=best, red=worst
QUINTILE_PALETTE = {
    "Q1 (Best)": "#2c7bb6",
    "Q2":        "#abd9e9",
    "Q3":        "#ffffbf",
    "Q4":        "#fdae61",
    "Q5 (Worst)": "#d7191c",
}

# Shift colormap: blue=improved, grey=same, red=worsened
SHIFT_COLORS = {
    -3: "#1a6faf", -2: "#3a9ad9", -1: "#8ec9e8",
     0: "#cccccc",
     1: "#f4a582",  2: "#d6604d",  3: "#a50026",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style():
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.05)
    plt.rcParams.update({
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "district_hadi.csv")
    df["dept_name"] = df["iddpto"].map(DEPT_NAMES).fillna(df["iddpto"].astype(str))
    return df


# ── Figure 1 — Facility supply distribution (Q1) ─────────────────────────────

def fig01_supply_distribution(df: pd.DataFrame):
    """
    Histogram of emergency facility counts per district.

    Q1 answered: Shows that supply is highly concentrated — the most important
    distributional fact is the mass at zero (44.5% of districts).

    Choice rationale: A histogram of the raw count reveals the distributional
    shape (extreme right skew, spike at 0) that a bar chart of individual
    districts would bury in noise. Log y-scale makes the long tail visible
    without distorting the dominant zero-count bar.
    Alternative rejected: choropleth map (Task 5); bar of top-N districts
    (ignores the full distribution and overstates individual differences).
    """
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))

    # Truncate display at 20 for readability (max is 47)
    plot_data = df["n_emergency_active"].clip(upper=20)
    bins = np.arange(-0.5, 21.5, 1)

    n_total = len(df)
    n_zero  = (df["n_emergency_active"] == 0).sum()

    counts, _, patches = ax.hist(
        plot_data, bins=bins,
        color="#4393c3", edgecolor="white", linewidth=0.6,
    )
    # Highlight the zero bar
    patches[0].set_facecolor("#d7191c")
    patches[0].set_alpha(0.85)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate zero bar
    ax.annotate(
        f"{n_zero:,} districts\n({n_zero/n_total:.1%})\nhave ZERO\nemergency\nfacilities",
        xy=(0, counts[0]), xytext=(3.5, counts[0] * 0.6),
        arrowprops=dict(arrowstyle="->", color="#d7191c"),
        color="#d7191c", fontsize=9, ha="left",
    )

    # Vertical lines at mean and median
    mn = df["n_emergency_active"].mean()
    md = df["n_emergency_active"].median()
    ax.axvline(mn, color="#333333", linestyle="--", linewidth=1.2,
               label=f"Mean = {mn:.1f}")
    ax.axvline(md, color="#555555", linestyle=":",  linewidth=1.5,
               label=f"Median = {int(md)}")

    ax.set_xlabel("SUSALUD-confirmed emergency facilities per district", fontsize=11)
    ax.set_ylabel("Number of districts (log scale)", fontsize=11)
    ax.set_title(
        "Q1 — Emergency facility supply is extremely concentrated\n"
        "(bars truncated at 20; max = 47)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(framealpha=0.8, fontsize=9)
    ax.set_xlim(-0.5, 20.5)

    _save(fig, "fig01_supply_distribution.png")


# ── Figure 2 — Facility supply vs emergency activity (Q1) ────────────────────

def fig02_supply_vs_activity(df: pd.DataFrame):
    """
    Scatter: log(n_emergency_active + 1) vs log(susalud_atenciones + 1).
    Points coloured by HADI quintile.

    Q1 answered: Tests whether having more facilities actually produces more
    emergency care (it does, but weakly — many high-activity districts have
    few registered facilities, revealing SUSALUD reporting gaps).

    Choice rationale: A scatter is the only chart that simultaneously shows
    both the facility dimension and the activity dimension of Q1, and reveals
    the relationship between them. Log1p transform compresses the extreme
    range (0–262k) while keeping zero-valued districts visible at the origin.
    Alternative rejected: two separate histograms (lose the joint relationship);
    a correlation coefficient alone (loses the heterogeneity of the cloud).
    """
    _style()
    fig, ax = plt.subplots(figsize=(9, 6))

    valid = df.dropna(subset=["hadi_quintile_baseline"])
    x = np.log1p(valid["n_emergency_active"])
    y = np.log1p(valid["susalud_atenciones"])

    for q in QUINTILE_ORDER:
        mask = valid["hadi_quintile_baseline"] == q
        ax.scatter(
            x[mask], y[mask],
            c=QUINTILE_PALETTE[q], label=q,
            s=18, alpha=0.65, edgecolors="none",
            zorder=3 if q in ("Q1 (Best)", "Q5 (Worst)") else 2,
        )

    # OLS trend line
    finite = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[finite], y[finite], 1)
    xr = np.array([x[finite].min(), x[finite].max()])
    ax.plot(xr, m * xr + b, color="#333333", linewidth=1.4,
            linestyle="--", label=f"OLS: slope={m:.2f}")

    # Axis labels with tick interpretation
    def log1p_fmt(val, _):
        real = np.expm1(val)
        if real < 1:
            return "0"
        elif real < 10:
            return f"{real:.0f}"
        elif real < 1000:
            return f"{real:.0f}"
        else:
            return f"{real/1000:.0f}k"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log1p_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log1p_fmt))

    ax.set_xlabel("Emergency facilities in district (log scale)", fontsize=11)
    ax.set_ylabel("SUSALUD emergency visits, latest year (log scale)", fontsize=11)
    ax.set_title(
        "Q1 — More facilities correlate with higher activity, but weakly\n"
        "(large cluster at origin: no facilities AND no recorded visits)",
        fontsize=12, fontweight="bold",
    )
    legend = ax.legend(
        title="HADI quintile",
        handles=[
            mpatches.Patch(color=QUINTILE_PALETTE[q], label=q)
            for q in QUINTILE_ORDER
        ] + [plt.Line2D([0], [0], color="#333333", linestyle="--",
                        label=f"OLS: slope={m:.2f}")],
        loc="upper left", fontsize=8, title_fontsize=9,
    )
    ax.add_artist(legend)

    _save(fig, "fig02_supply_vs_activity.png")


# ── Figure 3 — Spatial access (Q2) ───────────────────────────────────────────

def fig03_spatial_access(df: pd.DataFrame):
    """
    Two-panel spatial access analysis.

    Panel A: histogram of % CCPP > 20 km from nearest emergency facility.
    Panel B: horizontal box plots of median CCPP distance (km) by department,
             sorted by median — reveals geographic concentration of isolation.

    Q2 answered: Panel A shows that most districts are well-served (median=0%)
    but a long right tail includes districts where ALL populated centers are
    beyond 20 km. Panel B pins that isolation geographically: Loreto, Ucayali,
    Madre de Dios, and Amazonas dominate the worst distances.

    Choice rationale: The histogram alone shows the distribution shape; the
    department box plot adds the WHERE. Together they answer Q2 more fully than
    either alone.
    Alternative rejected: choropleth (Task 5); per-district sorted bar (1,873
    bars are unreadable and obscure the key finding of departmental clustering).
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ── Panel A: histogram of % CCPP > 20 km ─────────────────────────────────
    ax = axes[0]
    pct = df["pct_ccpp_gt20km_baseline"].dropna()

    ax.hist(pct, bins=30, color="#4393c3", edgecolor="white", linewidth=0.5)
    n_zero   = (pct == 0).sum()
    n_all    = (pct == 100).sum()
    n_half   = (pct >= 50).sum()

    ax.axvline(pct.mean(), color="#d7191c", linestyle="--", linewidth=1.3,
               label=f"Mean = {pct.mean():.1f}%")
    ax.axvline(pct.median(), color="#555555", linestyle=":", linewidth=1.5,
               label=f"Median = {pct.median():.0f}%")

    ax.text(55, ax.get_ylim()[1] * 0.85,
            f"{n_zero:,} districts:\n0% CCPP beyond 20 km",
            fontsize=8, color="#1a6faf", ha="left")
    ax.text(55, ax.get_ylim()[1] * 0.65,
            f"{n_all:,} districts:\nALL CCPP beyond 20 km",
            fontsize=8, color="#d7191c", ha="left")

    ax.set_xlabel("% of populated centers > 20 km from nearest facility", fontsize=10)
    ax.set_ylabel("Number of districts", fontsize=10)
    ax.set_title("A — National distribution of spatial isolation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Panel B: box plots by department ─────────────────────────────────────
    ax2 = axes[1]
    dept_data = df.dropna(subset=["dist_median_m_baseline"]).copy()
    dept_data["dist_median_km"] = dept_data["dist_median_m_baseline"] / 1000

    # Sort departments by median distance (worst at top)
    order = (
        dept_data.groupby("dept_name")["dist_median_km"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    sns.boxplot(
        data=dept_data,
        y="dept_name", x="dist_median_km",
        order=order,
        color="#abd9e9",
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
        linewidth=0.8,
        ax=ax2,
    )
    ax2.axvline(20, color="#d7191c", linestyle="--", linewidth=1.2,
                label="20 km threshold")
    ax2.set_xlabel("Median CCPP distance to nearest facility (km)", fontsize=10)
    ax2.set_ylabel("")
    ax2.set_title("B — Distance by department (sorted by median)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Q2 — Spatial access: a zero-inflated gap between well-served and isolated districts",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig03_spatial_access.png")


# ── Figure 4 — HADI distribution: baseline vs alternative (Q3 + Q4) ──────────

def fig04_hadi_distribution(df: pd.DataFrame):
    """
    Histogram of HADI (baseline) with quintile band shading, overlaid with
    KDE curves for baseline (solid) and alternative (dashed).

    Q3 answered: The histogram shows how districts spread across the deprivation
    spectrum; the mode in Q3 (0.4–0.6) is the "moderate" cluster.

    Q4 answered: The two KDE curves diverge notably above HADI=0.6, showing
    that the alternative specification re-classifies a subset of districts as
    more deprived — confirming the sensitivity finding (555 districts shift +1).

    Choice rationale: A dual-KDE overlay is the most direct way to compare two
    continuous distributions on the same scale without hiding their shapes.
    Alternative rejected: two separate histograms (harder to compare peaks);
    box-and-whisker summary alone (loses the bimodal shoulder visible in KDE).
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    valid = df.dropna(subset=["hadi_baseline", "hadi_alternative"])

    # Quintile band shading
    bands = [
        (0.0, 0.2, QUINTILE_PALETTE["Q1 (Best)"], "Q1 (Best)  [0.0–0.2]"),
        (0.2, 0.4, QUINTILE_PALETTE["Q2"],        "Q2  [0.2–0.4]"),
        (0.4, 0.6, QUINTILE_PALETTE["Q3"],        "Q3  [0.4–0.6]"),
        (0.6, 0.8, QUINTILE_PALETTE["Q4"],        "Q4  [0.6–0.8]"),
        (0.8, 1.0, QUINTILE_PALETTE["Q5 (Worst)"],"Q5 (Worst)  [0.8–1.0]"),
    ]
    for lo, hi, color, label in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.18, label=label)

    # HADI histogram (baseline counts)
    ax.hist(
        valid["hadi_baseline"], bins=40,
        color="#4393c3", alpha=0.35, density=True,
        edgecolor="white", linewidth=0.3, label="_nolegend_",
    )

    # KDE curves
    sns.kdeplot(
        valid["hadi_baseline"], ax=ax,
        color="#1f4e79", linewidth=2.2,
        label="Baseline (SUSALUD-confirmed)",
    )
    sns.kdeplot(
        valid["hadi_alternative"], ax=ax,
        color="#c0392b", linewidth=2.2, linestyle="--",
        label="Alternative (structural category ≥ I-3)",
    )

    ax.set_xlabel("HADI score (0 = best served, 1 = most deprived)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Q3 + Q4 — HADI distribution: most districts cluster in Q2–Q3;\n"
        "alternative specification shifts tail toward higher deprivation",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, 1)

    # Legend — bands first, then KDE lines
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=8.5,
              loc="upper right", ncol=1, framealpha=0.85)

    _save(fig, "fig04_hadi_distribution.png")


# ── Figure 5 — HADI components by quintile (Q3) ──────────────────────────────

def fig05_components_by_quintile(df: pd.DataFrame):
    """
    Grouped bar chart: mean HADI component score (Facility, Activity, Access)
    for each HADI quintile — baseline specification.

    Q3 answered: Reveals WHICH dimension of deprivation defines each quintile.
    Q5 districts score near 1.0 on all three components simultaneously, but the
    facility and activity components separate Q1 from Q3 more sharply than the
    access component does, indicating that structural supply is the dominant
    driver of the composite index.

    Choice rationale: A grouped bar is the clearest way to compare three
    categorical variables across an ordinal axis when the audience needs to
    read absolute values, not just rank order.
    Alternative rejected: radar/spider chart (distorts area perception);
    heatmap (harder to read exact values and requires colour interpretation).
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    valid = df.dropna(subset=["hadi_quintile_baseline"])
    comp_cols = {
        "Facility Density\n(comp_facility)": "comp_facility_baseline",
        "Emergency Activity\n(comp_activity)": "comp_activity_baseline",
        "Spatial Access\n(comp_access)": "comp_access_baseline",
    }

    means = (
        valid.groupby("hadi_quintile_baseline")[list(comp_cols.values())]
        .mean()
        .reindex(QUINTILE_ORDER)
        .rename(columns={v: k for k, v in comp_cols.items()})
        .reset_index()
        .rename(columns={"hadi_quintile_baseline": "Quintile"})
        .melt(id_vars="Quintile", var_name="Component", value_name="Mean score")
    )

    comp_palette = {
        "Facility Density\n(comp_facility)":  "#2c7bb6",
        "Emergency Activity\n(comp_activity)": "#d7191c",
        "Spatial Access\n(comp_access)":      "#1a9641",
    }

    sns.barplot(
        data=means,
        x="Quintile", y="Mean score",
        hue="Component",
        palette=comp_palette,
        order=QUINTILE_ORDER,
        width=0.7,
        ax=ax,
    )

    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1,
               label="National average (0.5 by construction)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("HADI quintile (baseline)", fontsize=11)
    ax.set_ylabel("Mean component score\n(0 = least deprived, 1 = most deprived)", fontsize=10)
    ax.set_title(
        "Q3 — What drives deprivation? All three components rise together,\n"
        "but facility density and activity separate Q1 from Q3 most sharply",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Component", fontsize=9, title_fontsize=9,
              loc="upper left", framealpha=0.85)

    _save(fig, "fig05_components_by_quintile.png")


# ── Figure 6 — Sensitivity: baseline vs alternative HADI (Q4) ────────────────

def fig06_sensitivity(df: pd.DataFrame):
    """
    Scatter: hadi_baseline (x) vs hadi_alternative (y).
    Points coloured by quintile_shift (-3 … +3).
    Reference diagonal y = x included.

    Q4 answered: The diagonal concentration (same specification) vs off-diagonal
    scatter (changed classification) quantifies how stable the index is.
    Most districts sit near y = x; the alternative shifts Q4 districts upward
    (toward more deprivation) because it uses a stricter facility set (1,854 vs
    3,093 facilities). Districts that improve are those where structural category
    captures facilities not confirmed in SUSALUD.

    Choice rationale: A scatter on the same scale with a y=x reference directly
    shows agreement vs disagreement between two continuous scores. Colouring by
    shift magnitude adds the quintile dimension without a separate panel.
    Alternative rejected: grouped bar of quintile-shift counts (loses the
    continuous HADI information and the spatial pattern of disagreement);
    confusion matrix / cross-tab alone (hides the direction of disagreement).
    """
    _style()
    fig, ax = plt.subplots(figsize=(8, 7))

    valid = df.dropna(subset=["hadi_baseline", "hadi_alternative", "quintile_shift"])
    valid["shift_int"] = valid["quintile_shift"].astype(int)

    for shift, group in valid.groupby("shift_int"):
        color = SHIFT_COLORS.get(shift, "#cccccc")
        n = len(group)
        label = (
            "same" if shift == 0
            else (f"worsened {abs(shift)}" if shift > 0 else f"improved {abs(shift)}")
        ) + f" (n={n:,})"
        ax.scatter(
            group["hadi_baseline"], group["hadi_alternative"],
            c=color, s=15, alpha=0.65, edgecolors="none", label=label,
            zorder=4 if shift != 0 else 2,
        )

    # Reference diagonal
    lo, hi = 0.05, 0.85
    ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.5,
            linestyle="--", label="y = x (no change)", zorder=5)

    # Quadrant annotations
    ax.text(0.1, 0.78,
            "Alternative: MORE deprived\n(above diagonal)",
            fontsize=8, color="#a50026", ha="left", style="italic")
    ax.text(0.5, 0.12,
            "Alternative: LESS deprived\n(below diagonal)",
            fontsize=8, color="#1a6faf", ha="left", style="italic")

    n_same = (valid["shift_int"] == 0).sum()
    n_total = len(valid)
    ax.set_title(
        f"Q4 — Sensitivity: {n_same:,}/{n_total:,} districts ({n_same/n_total:.0%}) unchanged;\n"
        "alternative (structural) classifies more districts as MORE deprived",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("HADI — baseline (SUSALUD-confirmed facilities)", fontsize=11)
    ax.set_ylabel("HADI — alternative (structural category ≥ I-3)", fontsize=11)
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.05, 0.85)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85, title="Quintile shift")

    _save(fig, "fig06_sensitivity.png")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_visualization_pipeline():
    """
    Generate all 6 static figures for Task 4.

    Reads  : output/tables/district_hadi.csv
    Writes : output/figures/fig01_*.png … fig06_*.png
    """
    print("\n=== Task 4 — Static Visualization ===\n")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = _load()
    print(f"  Loaded district_hadi: {len(df):,} rows\n")

    fig01_supply_distribution(df)
    fig02_supply_vs_activity(df)
    fig03_spatial_access(df)
    fig04_hadi_distribution(df)
    fig05_components_by_quintile(df)
    fig06_sensitivity(df)

    print(f"\n  All figures saved to {FIGURES_DIR}")



# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Geospatial maps (GeoPandas static + Folium interactive)
# ═══════════════════════════════════════════════════════════════════════════════

# Structural emergency categories (same definition as geospatial.py)
_EMRG_CATS = {"I-3", "I-4", "II-1", "II-2", "II-E", "III-1", "III-2", "III-E"}

# Categorical choropleth helpers
_Q_TO_INT = {"Q1 (Best)": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5 (Worst)": 5}
_Q_COLORS = ["#cccccc"] + [QUINTILE_PALETTE[q] for q in QUINTILE_ORDER]
_QCMAP    = ListedColormap(_Q_COLORS)
_QNORM    = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=6)

PERU_CENTER = [-9.19, -75.0]


def _save_html(m: folium.Map, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    m.save(str(path))
    print(f"  Saved -> {path.name}")


def _q_col(gdf: gpd.GeoDataFrame, quintile_col: str) -> pd.Series:
    """Map quintile string to integer code (0=no data, 1-5=Q1-Q5)."""
    return gdf[quintile_col].map(_Q_TO_INT).fillna(0).astype(float)


def _dept_boundaries(master: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Return dissolved department boundary lines for overlay."""
    return master.dissolve(by="iddpto").boundary


def _q_legend_patches(with_nodata: bool = False) -> list:
    patches = []
    if with_nodata:
        patches.append(MPatch(facecolor="#cccccc", edgecolor="#999", label="No data"))
    for q in QUINTILE_ORDER:
        patches.append(MPatch(facecolor=QUINTILE_PALETTE[q], edgecolor="#999", label=q))
    return patches


def _prep_for_folium(master: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Slim + simplify district GeoDataFrame for Folium export.
    Converts all nullable types, fills NaN, and simplifies geometry
    to ~1 km tolerance for faster HTML rendering.
    """
    cols = [
        "ubigeo", "iddpto",
        "hadi_baseline", "hadi_quintile_baseline",
        "hadi_alternative", "hadi_quintile_alternative",
        "quintile_shift", "classification_change",
        "n_emergency_active", "n_emergency_structural",
        "pct_ccpp_gt20km_baseline", "pct_ccpp_gt20km_alternative",
        "dist_median_m_baseline", "susalud_atenciones",
        "n_ccpp_baseline", "camas_total",
        "geometry",
    ]
    df = master[cols].copy()
    df.geometry = df.geometry.simplify(0.01, preserve_topology=True)

    # Round floats, fill NA → JSON-safe
    for c in ["hadi_baseline", "hadi_alternative"]:
        df[c] = df[c].round(3).fillna(-1)
    for c in ["pct_ccpp_gt20km_baseline", "pct_ccpp_gt20km_alternative"]:
        df[c] = df[c].round(1).fillna(-1)
    df["dist_median_km"] = (df["dist_median_m_baseline"] / 1000).round(1).fillna(-1)
    for c in ["n_ccpp_baseline", "camas_total", "n_emergency_active",
              "n_emergency_structural", "susalud_atenciones"]:
        df[c] = df[c].fillna(0).astype(int)
    df["quintile_shift"] = df["quintile_shift"].fillna(0).astype(int)
    for c in ["hadi_quintile_baseline", "hadi_quintile_alternative",
              "classification_change"]:
        df[c] = df[c].fillna("No data").astype(str).replace("nan", "No data")
    return df


# ── Static map 1 — HADI quintile choropleth (Q3) ─────────────────────────────

def map01_hadi_choropleth(master: gpd.GeoDataFrame):
    """
    GeoPandas choropleth of HADI baseline quintile across all 1,873 districts.

    Q3 answered spatially: reveals the strong geographic concentration of
    deprivation — Q5 districts cluster in Loreto, Ucayali, and Amazonas
    (Amazon basin), while Q1 districts concentrate along the coast and in
    Lima, showing a coastal–Amazonian divide that is not visible in charts.

    Choice rationale: A choropleth is the only chart type that can show BOTH
    which districts are deprived AND where they are simultaneously.
    Alternative rejected: a ranked bar of all 1,873 districts (unreadable);
    a dot density map (adds noise without adding geographic structure).
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 14))

    df = master.copy()
    df["_q"] = _q_col(df, "hadi_quintile_baseline")

    df.plot(column="_q", cmap=_QCMAP, norm=_QNORM, ax=ax,
            linewidth=0.08, edgecolor="#cccccc")
    _dept_boundaries(master).plot(ax=ax, color="white", linewidth=0.7, alpha=0.7)

    # Legend with district counts
    q_counts = master["hadi_quintile_baseline"].value_counts()
    patches = [MPatch(facecolor="#cccccc", edgecolor="#aaa", label="No data (3)")]
    for q in QUINTILE_ORDER:
        n = q_counts.get(q, 0)
        patches.append(MPatch(facecolor=QUINTILE_PALETTE[q], edgecolor="#999",
                               label=f"{q}  (n={n:,})"))

    ax.legend(handles=patches, loc="lower left", fontsize=9,
              title="HADI Quintile", title_fontsize=10, framealpha=0.9)
    ax.set_title(
        "Q3 — Healthcare Access Deprivation Index (Baseline)\n"
        "Higher quintile = more underserved. Department boundaries in white.",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axis("off")
    _save(fig, "map01_hadi_choropleth.png")


# ── Static map 2 — Baseline vs Alternative side-by-side (Q4) ─────────────────

def map02_baseline_vs_alternative(master: gpd.GeoDataFrame):
    """
    Side-by-side GeoPandas choropleth: baseline (left) vs alternative (right).

    Q4 answered spatially: The two maps use identical district geometries and
    identical colour scale; visual differences pin WHERE the specification
    choice matters most. The Amazon basin expands into Q4–Q5 under the
    alternative (fewer structural facilities), while the coast stays mostly Q1.

    Choice rationale: A direct side-by-side map is the clearest way to show
    spatial sensitivity — the reader can immediately scan for colour changes.
    Alternative rejected: a single 'difference' map (quintile-shift choropleth)
    loses the absolute deprivation level, showing only the delta. Both
    the absolute classification AND the change are analytically important.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 14))

    df = master.copy()
    df["_qb"] = _q_col(df, "hadi_quintile_baseline")
    df["_qa"] = _q_col(df, "hadi_quintile_alternative")
    dept_bounds = _dept_boundaries(master)

    for ax, col, title in [
        (axes[0], "_qb", "Baseline\n(SUSALUD-confirmed, 3,093 facilities)"),
        (axes[1], "_qa", "Alternative\n(structural category ≥ I-3, 1,854 facilities)"),
    ]:
        df.plot(column=col, cmap=_QCMAP, norm=_QNORM, ax=ax,
                linewidth=0.08, edgecolor="#cccccc")
        dept_bounds.plot(ax=ax, color="white", linewidth=0.7, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    patches = [MPatch(facecolor="#cccccc", edgecolor="#aaa", label="No data")]
    for q in QUINTILE_ORDER:
        patches.append(MPatch(facecolor=QUINTILE_PALETTE[q], edgecolor="#999", label=q))
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=10, title="HADI Quintile", title_fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Q4 — Sensitivity: which districts change HADI quintile\n"
        "when switching from SUSALUD-confirmed to structural facility definition?",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    _save(fig, "map02_baseline_vs_alternative.png")


# ── Static map 3 — Spatial access gap choropleth (Q2) ────────────────────────

def map03_access_gap(master: gpd.GeoDataFrame):
    """
    Continuous choropleth of % populated centers > 20 km from nearest
    emergency facility (baseline specification).

    Q2 answered spatially: Reveals that spatial isolation is a distinct
    phenomenon from structural supply shortage. Several districts with moderate
    HADI scores (Q3) have high isolation because their populated centers are
    scattered — a pattern visible only on a map, not in the charts.

    Choice rationale: A continuous graduated colour scale (0–100%) shows the
    full gradient of spatial isolation and avoids discretisation artefacts.
    White-to-red highlights the severity gradient naturally (white = no gap).
    Alternative rejected: binary map (above/below 50%) erases the gradient;
    the Q3 map (fig02 left panel) already shows the composite score.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 14))

    df = master.copy()
    df["pct20"] = df["pct_ccpp_gt20km_baseline"].fillna(-1)

    # Use custom bins: 0, 10, 25, 50, 75, 100
    df_valid = df[df["pct20"] >= 0]
    df_miss  = df[df["pct20"] < 0]

    df_miss.plot(ax=ax, color="#dddddd", linewidth=0.08, edgecolor="#cccccc")
    df_valid.plot(
        column="pct20", cmap="RdYlGn_r",
        vmin=0, vmax=100,
        ax=ax, linewidth=0.08, edgecolor="#cccccc",
        legend=True,
        legend_kwds={
            "label": "% of populated centers > 20 km from nearest facility",
            "orientation": "horizontal",
            "shrink": 0.55,
            "pad": 0.02,
        },
    )
    _dept_boundaries(master).plot(ax=ax, color="white", linewidth=0.7, alpha=0.7)

    n_zero = (df_valid["pct20"] == 0).sum()
    n_all  = (df_valid["pct20"] == 100).sum()
    ax.text(
        0.02, 0.04,
        f"Deep green = 0% isolated ({n_zero:,} districts)\n"
        f"Deep red = 100% isolated ({n_all:,} districts)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
    )

    ax.set_title(
        "Q2 — Spatial Access Gap: % of populated centers\n"
        "more than 20 km from the nearest emergency facility (baseline)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.axis("off")
    _save(fig, "map03_access_gap.png")


# ── Interactive map 4 — HADI explorer (Q3 + Q4) ──────────────────────────────

def map04_hadi_explorer(master: gpd.GeoDataFrame):
    """
    Folium choropleth with baseline and alternative HADI layers (toggleable).

    Features
    --------
    - Layer 1 (default): baseline HADI quintile with custom colours
    - Layer 2 (toggle):  alternative HADI quintile — same scale, different values
    - Hover tooltip: UBIGEO, quintile, HADI score, facilities, distance, SUSALUD visits
    - Layer control to switch specifications
    - CartoDB Positron basemap (minimal, keeps district colours prominent)

    Q3 + Q4 answered interactively: the reader can hover over any district to
    read all six underlying metrics, and toggle between specifications to observe
    reclassification in context — something static maps cannot provide.
    """
    gdf = _prep_for_folium(master)
    geojson_str = gdf.to_json()

    m = folium.Map(location=PERU_CENTER, zoom_start=5,
                   tiles="CartoDB positron", control_scale=True)

    tooltip_fields  = ["ubigeo", "hadi_quintile_baseline", "hadi_baseline",
                       "n_emergency_active", "pct_ccpp_gt20km_baseline",
                       "dist_median_km", "susalud_atenciones", "camas_total"]
    tooltip_aliases = ["UBIGEO:", "Quintile (baseline):", "HADI score:",
                       "Emergency facilities:", "% CCPP > 20 km:",
                       "Median dist. (km):", "SUSALUD visits:", "Beds (camas):"]

    # Layer 1 — baseline (shown by default)
    folium.GeoJson(
        data=geojson_str,
        style_function=lambda feat: {
            "fillColor": QUINTILE_PALETTE.get(
                feat["properties"].get("hadi_quintile_baseline", ""), "#cccccc"),
            "fillOpacity": 0.75,
            "color": "white",
            "weight": 0.4,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            style="font-size:12px;",
        ),
        name="HADI — Baseline (SUSALUD-confirmed)",
        show=True,
    ).add_to(m)

    # Layer 2 — alternative (hidden by default, toggled via LayerControl)
    alt_tooltip_fields  = ["ubigeo", "hadi_quintile_alternative", "hadi_alternative",
                            "n_emergency_structural", "pct_ccpp_gt20km_alternative",
                            "dist_median_km", "susalud_atenciones"]
    alt_tooltip_aliases = ["UBIGEO:", "Quintile (alternative):", "HADI score:",
                            "Structural facilities:", "% CCPP > 20 km:",
                            "Median dist. (km):", "SUSALUD visits:"]
    folium.GeoJson(
        data=geojson_str,
        style_function=lambda feat: {
            "fillColor": QUINTILE_PALETTE.get(
                feat["properties"].get("hadi_quintile_alternative", ""), "#cccccc"),
            "fillOpacity": 0.75,
            "color": "white",
            "weight": 0.4,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=alt_tooltip_fields,
            aliases=alt_tooltip_aliases,
            localize=True, sticky=True,
            style="font-size:12px;",
        ),
        name="HADI — Alternative (structural category ≥ I-3)",
        show=False,
    ).add_to(m)

    # HTML legend (Folium GeoJson has no built-in legend)
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:10px 14px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,.3);
         font-family:sans-serif;font-size:12px;">
      <b>HADI Quintile</b><br>
      <i style="background:#2c7bb6;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;"></i>Q1 (Best) &nbsp;
      <i style="background:#abd9e9;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;"></i>Q2<br>
      <i style="background:#ffffbf;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;border:1px solid #ccc;"></i>Q3 &nbsp;
      <i style="background:#fdae61;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;"></i>Q4<br>
      <i style="background:#d7191c;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;"></i>Q5 (Worst) &nbsp;
      <i style="background:#cccccc;width:14px;height:14px;display:inline-block;
         border-radius:2px;margin-right:5px;"></i>No data<br>
      <small>Toggle layers with the &#x2261; control (top-right)</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    _save_html(m, "map_hadi_explorer.html")


# ── Interactive map 5 — Facilities on access-gap background (Q1 + Q2) ─────────

def map05_facilities_access(master: gpd.GeoDataFrame,
                             ipress: gpd.GeoDataFrame,
                             facility_annual: pd.DataFrame):
    """
    Folium map: district access gap + emergency facility markers.

    Layers
    ------
    - Background districts: pct_ccpp_gt20km_baseline choropleth (Red-Green)
    - Baseline facilities (SUSALUD-confirmed, blue markers)
    - Alternative facilities not in baseline (structural-only, orange markers)
    - Both marker layers are toggleable via LayerControl

    Q1 + Q2 answered interactively: shows WHERE facilities exist in the context
    of spatial isolation, exposing coverage gaps — districts with high isolation
    AND few/no markers are the hardest to serve. The two marker layers reveal
    which additional facilities the structural definition would include.

    Marker tooltip shows facility name, category, institution type.
    District tooltip shows % isolation and district UBIGEO.
    """
    gdf = _prep_for_folium(master)

    # Continuous colormap for pct_ccpp_gt20km (0-100)
    pct_cmap = bcm.LinearColormap(
        colors=["#1a9641", "#ffffbf", "#d7191c"],
        vmin=0, vmax=100,
        caption="% populated centers > 20 km from nearest emergency facility",
    )

    m = folium.Map(location=PERU_CENTER, zoom_start=5,
                   tiles="CartoDB positron", control_scale=True)

    # ── District background ───────────────────────────────────────────────────
    folium.GeoJson(
        data=gdf.to_json(),
        style_function=lambda feat: {
            "fillColor": (
                pct_cmap(feat["properties"]["pct_ccpp_gt20km_baseline"])
                if feat["properties"]["pct_ccpp_gt20km_baseline"] >= 0
                else "#dddddd"
            ),
            "fillOpacity": 0.65,
            "color": "white",
            "weight": 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ubigeo", "pct_ccpp_gt20km_baseline", "n_emergency_active"],
            aliases=["UBIGEO:", "% CCPP > 20 km:", "Emergency facilities (baseline):"],
            sticky=True, style="font-size:12px;",
        ),
        name="District — % CCPP > 20 km (spatial access gap)",
        show=True,
    ).add_to(m)
    pct_cmap.add_to(m)

    # ── Emergency facility markers ────────────────────────────────────────────
    confirmed_ids  = set(facility_annual["co_ipress"].unique())
    ipress_valid   = ipress[ipress["coords_valid"]].copy()
    baseline_fac   = ipress_valid[ipress_valid["codigo_unico"].isin(confirmed_ids)]
    alt_only_fac   = ipress_valid[
        ipress_valid["categoria"].isin(_EMRG_CATS)
        & ~ipress_valid["codigo_unico"].isin(confirmed_ids)
    ]

    def _add_markers(fac_gdf: gpd.GeoDataFrame, layer_name: str,
                     color: str, fill_color: str, show: bool):
        fg = folium.FeatureGroup(name=layer_name, show=show)
        for _, row in fac_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=4,
                color=color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.75,
                weight=1,
                tooltip=folium.Tooltip(
                    f"<b>{row['nombre'][:40]}</b><br>"
                    f"Category: {row['categoria']}<br>"
                    f"Type: {row['institucion']}<br>"
                    f"UBIGEO: {row['ubigeo']}",
                    sticky=True,
                ),
            ).add_to(fg)
        fg.add_to(m)

    _add_markers(baseline_fac, "Baseline facilities (SUSALUD-confirmed)",
                 "#1f4e79", "#4393c3", show=True)
    _add_markers(alt_only_fac, "Alt-only facilities (structural, not in SUSALUD)",
                 "#7f3300", "#fdae61", show=False)

    folium.LayerControl(collapsed=False).add_to(m)
    _save_html(m, "map_facilities_access.html")


# ── Maps pipeline ─────────────────────────────────────────────────────────────

def run_maps_pipeline():
    """
    Generate all 5 geospatial maps for Task 5.

    Static   : map01_hadi_choropleth.png  (Q3)
               map02_baseline_vs_alternative.png  (Q4)
               map03_access_gap.png  (Q2)
    Interactive: map_hadi_explorer.html  (Q3 + Q4)
                 map_facilities_access.html  (Q1 + Q2)

    Reads  : data/processed/district_master.gpkg
             data/processed/ipress_clean.gpkg
             data/processed/susalud_facility_annual.parquet
    Writes : output/figures/map0*.png
             output/figures/map_*.html
    """
    print("\n=== Task 5 — Geospatial Maps ===\n")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading data...")
    master          = gpd.read_file(PROCESSED_DIR / "district_master.gpkg")
    ipress          = gpd.read_file(PROCESSED_DIR / "ipress_clean.gpkg")
    facility_annual = pd.read_parquet(PROCESSED_DIR / "susalud_facility_annual.parquet")
    print(f"  Master: {len(master):,} districts | IPRESS: {len(ipress):,}\n")

    print("-- Static maps --")
    map01_hadi_choropleth(master)
    map02_baseline_vs_alternative(master)
    map03_access_gap(master)

    print("\n-- Interactive maps --")
    map04_hadi_explorer(master)
    map05_facilities_access(master, ipress, facility_annual)

    print(f"\n  All maps saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_visualization_pipeline()
    run_maps_pipeline()
