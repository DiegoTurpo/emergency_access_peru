"""
Task 3 — District-level Healthcare Access Deprivation Index (HADI).

Research questions answered
----------------------------
Q1. Which districts have lower/higher facility availability and emergency
    care activity?
    → comp_facility_{label}, comp_activity_{label}, hadi_{label}

Q2. Which districts have populated centers with weaker spatial access?
    → comp_access_{label}, pct_ccpp_gt20km_{label}, dist_median_m_{label}

Q3. Which districts are most underserved vs best served overall?
    → hadi_{label}, hadi_quintile_{label}  (Q5 = most deprived)

Q4. Sensitivity: baseline vs alternative comparison.
    → quintile_shift, classification_change, metrics_report.md

Index design
------------
HADI (Healthcare Access Deprivation Index) in [0, 1], higher = more deprived.
Three equal-weight components, each ranked to [0, 1] via percentile rank:

  Comp 1 — Facility Density (facility-related)
      = n_emergency_{label} / n_ccpp_{label} x 100  [facilities per 100 CCPP]
      Inverted: fewer facilities => higher deprivation

  Comp 2 — Emergency Activity (SUSALUD-based)
      = susalud_atenciones / n_ccpp_{label}  [emergency visits per CCPP]
      Inverted: fewer recorded visits => higher deprivation
      Same SUSALUD source in both versions; only denominator (n_ccpp) differs.

  Comp 3 — Spatial Access (populated-center access)
      = pct_ccpp_gt20km_{label}  [% of CCPP > 20 km from nearest facility]
      Direct: higher % => higher deprivation

Normalization
-------------
Percentile rank -> linear rescale to [0, 1].
Robust to heavy tails (median activity is 0; max is 262,245 — min-max would
compress 99% of districts into a tiny band near 0).
Ties receive their average rank.

Baseline   : n_emergency_active    (SUSALUD-confirmed facilities, ~3,093)
Alternative: n_emergency_structural (category >= I-3 facilities, ~1,854)
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_DIR    = Path(__file__).resolve().parents[1] / "output" / "tables"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Percentile rank normalized to [0, 1].
    ascending=True  -> largest value gets rank 1 (direct deprivation).
    ascending=False -> largest value gets rank 0 (inverted for supply).
    NaN values are preserved.
    """
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    n = valid.sum()
    if n < 2:
        return result
    ranks = series[valid].rank(method="average", ascending=ascending)
    result[valid] = (ranks - 1) / (n - 1)
    return result


def _to_quintile_int(q_series: pd.Series) -> pd.Series:
    """Extract integer 1-5 from quintile label strings like 'Q1 (Best)'."""
    return (
        q_series.astype(str)
        .str.extract(r"Q(\d)", expand=False)
        .astype(float)
    )


# ── Component & index computation ─────────────────────────────────────────────

def compute_hadi(master: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """
    Add HADI components and composite score for one specification.

    Parameters
    ----------
    label : "baseline" or "alternative"

    Columns added
    -------------
    facility_density_{label}   raw input to component 1
    activity_per_ccpp_{label}  raw input to component 2
    comp_facility_{label}      percentile rank [0,1], higher = more deprived
    comp_activity_{label}      percentile rank [0,1], higher = more deprived
    comp_access_{label}        percentile rank [0,1], higher = more deprived
    hadi_{label}               mean of 3 components [0,1]
    hadi_quintile_{label}      Q1 (Best) ... Q5 (Worst) -- equal-frequency bins
    """
    df = master.copy()

    n_emg_col  = "n_emergency_active" if label == "baseline" else "n_emergency_structural"
    n_ccpp_col = f"n_ccpp_{label}"
    pct_col    = f"pct_ccpp_gt20km_{label}"

    # Replace 0 CCPP with NaN to avoid division by zero
    n_ccpp = df[n_ccpp_col].astype(float).replace(0, np.nan)

    # ── Component 1: Facility Density ─────────────────────────────────────────
    df[f"facility_density_{label}"] = (
        df[n_emg_col].astype(float) / n_ccpp * 100
    )
    comp1 = _prank(df[f"facility_density_{label}"], ascending=False)

    # ── Component 2: Emergency Activity ──────────────────────────────────────
    df[f"activity_per_ccpp_{label}"] = (
        df["susalud_atenciones"].astype(float) / n_ccpp
    )
    comp2 = _prank(df[f"activity_per_ccpp_{label}"], ascending=False)

    # ── Component 3: Spatial Access ───────────────────────────────────────────
    comp3 = _prank(df[pct_col].astype(float), ascending=True)

    # ── Composite ─────────────────────────────────────────────────────────────
    components = pd.DataFrame({
        f"comp_facility_{label}": comp1,
        f"comp_activity_{label}": comp2,
        f"comp_access_{label}":   comp3,
    })

    df = pd.concat([df, components], axis=1)
    df[f"hadi_{label}"] = components.mean(axis=1, skipna=True).round(4)

    # ── Quintile classification (equal-interval score bands) ─────────────────
    # Using fixed [0, 0.2, 0.4, 0.6, 0.8, 1.0] bins so that the label
    # corresponds to an actual HADI range, not just a rank bucket.
    # Equal-frequency (qcut) produces very unequal bins when many districts
    # tie at 0 facilities / 0 activity — a known artifact of ordinal ranking.
    has_hadi = df[f"hadi_{label}"].notna()
    df[f"hadi_quintile_{label}"] = pd.NA

    if has_hadi.sum() >= 5:
        df.loc[has_hadi, f"hadi_quintile_{label}"] = pd.cut(
            df.loc[has_hadi, f"hadi_{label}"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.001],
            labels=["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"],
            include_lowest=True,
        ).astype(str)

    q_counts = df[f"hadi_quintile_{label}"].value_counts().sort_index()
    print(f"\n  HADI {label} quintile distribution:")
    for q, n in q_counts.items():
        print(f"    {q}: {n:,} districts")

    return df


def compute_comparison(master: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add comparison columns between baseline and alternative quintile assignments.

    Columns added
    -------------
    quintile_shift       : int  (alt - base); positive = more deprived under alt
    classification_change: str  "same" / "improved N" / "worsened N"
    """
    qi_base = _to_quintile_int(master["hadi_quintile_baseline"])
    qi_alt  = _to_quintile_int(master["hadi_quintile_alternative"])
    shift   = (qi_alt - qi_base).astype("Int64")

    master = master.copy()
    master["quintile_shift"] = shift

    def _label(x):
        if pd.isna(x):
            return pd.NA
        x = int(x)
        if x == 0:
            return "same"
        direction = "worsened" if x > 0 else "improved"
        return f"{direction} {abs(x)}"

    master["classification_change"] = shift.map(_label)

    print("\n  Baseline -> Alternative quintile shift distribution:")
    vc = shift.value_counts().sort_index()
    for v, n in vc.items():
        tag = "same" if v == 0 else (f"worsened {abs(v)}" if v > 0 else f"improved {abs(v)}")
        print(f"    {v:+d} ({tag}): {n:,} districts")

    return master


# ── Report generation ─────────────────────────────────────────────────────────

def _fmt(x, fmt=".1f"):
    return "—" if pd.isna(x) else format(float(x), fmt)


def save_metrics_report(master: gpd.GeoDataFrame):
    """Write a human-readable Markdown metrics report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("# HADI — Healthcare Access Deprivation Index\n")
    lines.append("**Higher score = more underserved**  |  Q5 = most deprived\n")
    lines.append("")

    lines.append("## National summary\n")
    lines.append("| Metric | Baseline | Alternative |")
    lines.append("|--------|----------|-------------|")
    for col_tmpl, row_label in [
        ("hadi_{label}",            "HADI mean"),
        ("comp_facility_{label}",   "Component 1 — Facility Density (mean)"),
        ("comp_activity_{label}",   "Component 2 — Emergency Activity (mean)"),
        ("comp_access_{label}",     "Component 3 — Spatial Access (mean)"),
    ]:
        b = master[col_tmpl.format(label="baseline")].mean()
        a = master[col_tmpl.format(label="alternative")].mean()
        lines.append(f"| {row_label} | {_fmt(b)} | {_fmt(a)} |")

    no_fac_b = (master["n_emergency_active"] == 0).sum()
    no_fac_a = (master["n_emergency_structural"] == 0).sum()
    lines.append(f"| Districts with zero emergency facility | {no_fac_b:,} | {no_fac_a:,} |")

    q5_b = (master["hadi_quintile_baseline"] == "Q5 (Worst)").sum()
    q5_a = (master["hadi_quintile_alternative"] == "Q5 (Worst)").sum()
    lines.append(f"| Districts in Q5 (most deprived) | {q5_b:,} | {q5_a:,} |")
    lines.append("")

    lines.append("## Quintile distribution\n")
    lines.append("| Quintile | Baseline (n) | Alternative (n) |")
    lines.append("|----------|--------------|-----------------|")
    for q in ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"]:
        nb = (master["hadi_quintile_baseline"] == q).sum()
        na = (master["hadi_quintile_alternative"] == q).sum()
        lines.append(f"| {q} | {nb:,} | {na:,} |")
    lines.append("")

    lines.append("## Sensitivity: baseline -> alternative quintile shifts\n")
    lines.append("| Shift | Districts | Interpretation |")
    lines.append("|-------|-----------|----------------|")
    vc = master["quintile_shift"].value_counts().sort_index()
    for v, n in vc.items():
        if pd.isna(v):
            continue
        v = int(v)
        if v == 0:
            interp = "no change"
        elif v > 0:
            interp = f"more deprived under alternative ({abs(v)} quintile{'s' if abs(v)>1 else ''})"
        else:
            interp = f"less deprived under alternative ({abs(v)} quintile{'s' if abs(v)>1 else ''})"
        lines.append(f"| {v:+d} | {n:,} | {interp} |")
    lines.append("")

    lines.append("## Top 15 most deprived districts (Q5, baseline)\n")
    q5 = (
        master[master["hadi_quintile_baseline"] == "Q5 (Worst)"]
        .sort_values("hadi_baseline", ascending=False)
        .head(15)
    )
    lines.append("| UBIGEO | HADI | Fac/100CCPP | Visits/CCPP | %CCPP>20km | Emg.Active |")
    lines.append("|--------|------|-------------|-------------|------------|------------|")
    for _, r in q5.iterrows():
        lines.append(
            f"| {r['ubigeo']} "
            f"| {_fmt(r['hadi_baseline'],'.3f')} "
            f"| {_fmt(r.get('facility_density_baseline'), '.2f')} "
            f"| {_fmt(r.get('activity_per_ccpp_baseline'), '.1f')} "
            f"| {_fmt(r['pct_ccpp_gt20km_baseline'], '.1f')} "
            f"| {int(r['n_emergency_active'])} |"
        )
    lines.append("")

    path = OUTPUT_DIR / "metrics_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {path}")


def save_quintile_csv(master: gpd.GeoDataFrame):
    """Flat CSV with all district metrics for downstream Streamlit use."""
    keep = [
        "ubigeo", "iddpto", "idprov",
        "n_facilities_total", "n_emergency_active", "n_emergency_structural",
        "camas_total", "n_public", "n_private",
        "n_ccpp_baseline",
        "dist_median_m_baseline", "dist_median_m_alternative",
        "pct_ccpp_gt5km_baseline",  "pct_ccpp_gt20km_baseline",
        "pct_ccpp_gt5km_alternative", "pct_ccpp_gt20km_alternative",
        "susalud_latest_year", "susalud_atenciones", "susalud_atendidos",
        "facility_density_baseline",   "activity_per_ccpp_baseline",
        "comp_facility_baseline", "comp_activity_baseline", "comp_access_baseline",
        "hadi_baseline", "hadi_quintile_baseline",
        "facility_density_alternative",  "activity_per_ccpp_alternative",
        "comp_facility_alternative", "comp_activity_alternative", "comp_access_alternative",
        "hadi_alternative", "hadi_quintile_alternative",
        "quintile_shift", "classification_change",
    ]
    cols = [c for c in keep if c in master.columns]
    out_path = OUTPUT_DIR / "district_hadi.csv"
    master[cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved -> {out_path}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_metrics_pipeline() -> gpd.GeoDataFrame:
    """
    Compute HADI for baseline and alternative, compare, and save outputs.

    Reads  : data/processed/district_master.gpkg
    Writes : data/processed/district_master.gpkg  (enriched, +14 columns)
             output/tables/metrics_report.md
             output/tables/district_hadi.csv
    """
    print("\n=== Task 3 — District Metrics ===\n")

    master = gpd.read_file(PROCESSED_DIR / "district_master.gpkg")
    print(f"  Loaded district_master: {len(master):,} rows x {len(master.columns)} cols")

    # Drop any HADI columns from a previous run to avoid duplicates on re-run
    hadi_prefixes = (
        "facility_density_", "activity_per_ccpp_",
        "comp_facility_", "comp_activity_", "comp_access_",
        "hadi_", "quintile_shift", "classification_change",
    )
    drop_cols = [c for c in master.columns if c.startswith(hadi_prefixes)]
    if drop_cols:
        master = master.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} stale HADI columns before recomputing")

    print("\n-- Baseline (SUSALUD-confirmed facilities) --")
    master = compute_hadi(master, "baseline")

    print("\n-- Alternative (structural category >= I-3) --")
    master = compute_hadi(master, "alternative")

    print("\n-- Sensitivity comparison --")
    master = compute_comparison(master)

    print("\n-- Saving outputs --")
    master.to_file(PROCESSED_DIR / "district_master.gpkg", driver="GPKG")
    print(f"  Saved -> district_master.gpkg  ({len(master.columns)} columns)")

    save_metrics_report(master)
    save_quintile_csv(master)

    print("\n=== Metrics Summary ===")
    for label in ["baseline", "alternative"]:
        h = master[f"hadi_{label}"]
        q5 = (master[f"hadi_quintile_{label}"] == "Q5 (Worst)").sum()
        print(f"  HADI {label}: mean={h.mean():.3f}  median={h.median():.3f}  "
              f"Q5 districts: {q5:,}")

    same    = (master["quintile_shift"] == 0).sum()
    changed = master["quintile_shift"].notna().sum() - same
    print(f"  Quintile unchanged: {same:,}  |  Changed: {changed:,}")
    print()

    return master


if __name__ == "__main__":
    run_metrics_pipeline()
