"""
Task 2 — Geospatial Integration with GeoPandas.

CRS Strategy
------------
EPSG:4326  (WGS-84, geographic, degrees)
    · All raw and processed data is stored in this CRS.
    · Used for display, Folium maps, and GeoPackage outputs.

EPSG:24891 (PSAD56 / Peru Central Zone, projected, meters)
    · Peru's official projected CRS — minimises distance/area distortion.
    · Used ONLY for metric calculations: distances, areas, centroids.
    · All results are converted back to EPSG:4326 before saving.

Spatial Operations Performed
-----------------------------
1. Assign IPRESS facilities to districts      (point-in-polygon, EPSG:4326)
2. Assign CCPP populated centers to districts (point-in-polygon, EPSG:4326)
3. Compute distance from each CCPP to nearest emergency IPRESS (EPSG:24891)
4. Aggregate to district-level supply and access metrics
5. Build a single district_master GeoDataFrame for Tasks 3-6
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CRS_GEO    = "EPSG:4326"    # storage / display
CRS_METRIC = "EPSG:24891"   # metric calculations (Peru UTM, metres)

# Categories considered structurally capable of emergency care
EMERGENCY_CATEGORIES = {"I-3", "I-4", "II-1", "II-2", "II-E", "III-1", "III-2", "III-E"}


# ── Loaders ───────────────────────────────────────────────────────────────────

def _gpkg(name: str) -> gpd.GeoDataFrame:
    return gpd.read_file(PROCESSED_DIR / f"{name}.gpkg")

def _parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


# ── Emergency facility subsets ────────────────────────────────────────────────

def get_emergency_ipress_active(
    ipress: gpd.GeoDataFrame,
    facility_annual: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    BASELINE definition — facilities confirmed to have reported emergency
    production to SUSALUD AND that have valid GPS coordinates.

    Rationale: SUSALUD Consulta-C1 records actual emergency care delivered,
    so this set captures real operational capacity, not just registration.
    """
    confirmed_ids = facility_annual["co_ipress"].unique()
    mask = (
        ipress["codigo_unico"].isin(confirmed_ids)
        & ipress["coords_valid"]
    )
    subset = ipress[mask].copy()
    print(f"  Emergency IPRESS (baseline — SUSALUD-confirmed + coords): {len(subset):,}")
    return subset


def get_emergency_ipress_structural(ipress: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    ALTERNATIVE definition — facilities whose registered category implies
    emergency capability (I-3 and above), regardless of SUSALUD reporting.

    Rationale: Some facilities provide emergency care but do not appear in
    SUSALUD due to reporting gaps. Using structural category is a broader
    proxy that may overcount but avoids under-counting silent reporters.
    """
    mask = ipress["categoria"].isin(EMERGENCY_CATEGORIES) & ipress["coords_valid"]
    subset = ipress[mask].copy()
    print(f"  Emergency IPRESS (alternative — structural category ≥ I-3): {len(subset):,}")
    return subset


# ── Spatial join: assign points to districts ─────────────────────────────────

def assign_to_districts(
    gdf_points: gpd.GeoDataFrame,
    gdf_districts: gpd.GeoDataFrame,
    label: str = "points",
) -> gpd.GeoDataFrame:
    """
    Assign each point to its containing district polygon.

    Method
    ------
    1. Spatial join with predicate='within' for clean assignments.
    2. sjoin_nearest fallback for the small number of points that fall
       on borders or just outside polygons due to coordinate precision.

    Both datasets must be in the same CRS (EPSG:4326).
    The result has a 'ubigeo_distrito' column with the assigned district.
    Existing 'ubigeo' in the points is preserved as 'ubigeo_original'.
    """
    assert gdf_points.crs == gdf_districts.crs, "CRS mismatch before spatial join"

    dist_cols = gdf_districts[["ubigeo", "iddpto", "idprov", "geometry"]].copy()
    dist_cols = dist_cols.rename(columns={
        "ubigeo": "ubigeo_distrito",
        "iddpto": "iddpto_distrito",
        "idprov": "idprov_distrito",
    })

    # Preserve original ubigeo if present
    if "ubigeo" in gdf_points.columns:
        gdf_points = gdf_points.rename(columns={"ubigeo": "ubigeo_original"})

    # Step 1: within join
    joined = gpd.sjoin(
        gdf_points, dist_cols, how="left", predicate="within"
    ).drop(columns="index_right", errors="ignore")

    # Step 2: nearest fallback for unmatched rows
    unmatched = joined["ubigeo_distrito"].isna()
    n_unmatched = unmatched.sum()
    if n_unmatched > 0:
        pts_unmatched = gdf_points[unmatched].drop(
            columns=["ubigeo_distrito", "iddpto_distrito", "idprov_distrito"],
            errors="ignore",
        )
        # Reproject to metric CRS for correct nearest-polygon search
        pts_proj  = pts_unmatched.to_crs(CRS_METRIC)
        dist_proj = dist_cols.to_crs(CRS_METRIC)
        fallback = gpd.sjoin_nearest(
            pts_proj,
            dist_proj,
            how="left",
            max_distance=None,
        ).drop(columns="index_right", errors="ignore")
        # Handle ties (keep first match per point)
        fallback = fallback[~fallback.index.duplicated(keep="first")]
        joined.loc[unmatched, ["ubigeo_distrito", "iddpto_distrito", "idprov_distrito"]] = (
            fallback[["ubigeo_distrito", "iddpto_distrito", "idprov_distrito"]].values
        )

    n_assigned = joined["ubigeo_distrito"].notna().sum()
    print(f"  {label}: {n_assigned:,} / {len(joined):,} assigned to a district "
          f"({n_unmatched:,} via nearest fallback)")
    return joined


# ── District supply metrics ───────────────────────────────────────────────────

def compute_district_supply(
    ipress: gpd.GeoDataFrame,
    facility_annual: pd.DataFrame,
    distritos: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Aggregate IPRESS facilities to district level.

    For each district this produces:
    - n_facilities_total      : registered IPRESS with valid coords
    - n_emergency_active      : confirmed in SUSALUD emergency data
    - n_emergency_structural  : category ≥ I-3
    - camas_total             : total inpatient beds
    - n_public                : MINSA + ESSALUD + GOBIERNO REGIONAL count
    - n_private               : PRIVADO count
    - n_cat_I                 : primary-care level (I-1 … I-4)
    - n_cat_II_plus           : hospital level (II-1 and above)
    """
    # Only facilities with valid coordinates can be spatially assigned
    ipress_coords = ipress[ipress["coords_valid"]].copy()

    # Assign to districts
    ipress_in_dist = assign_to_districts(ipress_coords, distritos, label="IPRESS")

    # Flags
    confirmed_ids = set(facility_annual["co_ipress"].unique())
    ipress_in_dist["is_emergency_active"]      = ipress_in_dist["codigo_unico"].isin(confirmed_ids)
    ipress_in_dist["is_emergency_structural"]  = ipress_in_dist["categoria"].isin(EMERGENCY_CATEGORIES)
    ipress_in_dist["is_public"] = ipress_in_dist["institucion"].isin(
        {"MINSA", "ESSALUD", "GOBIERNO REGIONAL",
         "MUNICIPALIDAD DISTRITAL", "MUNICIPALIDAD PROVINCIAL"}
    )
    ipress_in_dist["is_private"]  = ipress_in_dist["institucion"] == "PRIVADO"
    ipress_in_dist["is_cat_I"]    = ipress_in_dist["categoria"].str.startswith("I-")
    ipress_in_dist["is_cat_II_plus"] = ipress_in_dist["categoria"].str.startswith(("II-", "III-"))

    supply = (
        ipress_in_dist.groupby("ubigeo_distrito")
        .agg(
            n_facilities_total     = ("codigo_unico",            "count"),
            n_emergency_active     = ("is_emergency_active",     "sum"),
            n_emergency_structural = ("is_emergency_structural", "sum"),
            camas_total            = ("camas",                   "sum"),
            n_public               = ("is_public",               "sum"),
            n_private              = ("is_private",              "sum"),
            n_cat_I                = ("is_cat_I",                "sum"),
            n_cat_II_plus          = ("is_cat_II_plus",          "sum"),
        )
        .reset_index()
        .rename(columns={"ubigeo_distrito": "ubigeo"})
    )

    # Fill districts with zero facilities
    all_ubigeos = distritos[["ubigeo"]].copy()
    supply = all_ubigeos.merge(supply, on="ubigeo", how="left").fillna(0)
    int_cols = [c for c in supply.columns if c != "ubigeo"]
    supply[int_cols] = supply[int_cols].astype(int)

    print(f"  District supply: {len(supply):,} districts | "
          f"{supply['n_facilities_total'].sum():,} facilities assigned")
    return supply


# ── CCPP nearest-facility distance ────────────────────────────────────────────

def compute_nearest_facility(
    ccpp: gpd.GeoDataFrame,
    ipress_emergency: gpd.GeoDataFrame,
    label: str = "baseline",
) -> gpd.GeoDataFrame:
    """
    For each populated center compute the straight-line distance (metres)
    to the nearest emergency-capable facility.

    CRS: both inputs are reprojected to EPSG:24891 (Peru UTM) before
    distance computation so results are in metres, then converted back.

    Uses geopandas sjoin_nearest with spatial indexing for performance.
    """
    ccpp_proj    = ccpp[["geometry"]].to_crs(CRS_METRIC).copy()
    ipress_proj  = ipress_emergency[["codigo_unico", "nombre", "categoria", "geometry"]].to_crs(CRS_METRIC).copy()

    nearest = gpd.sjoin_nearest(
        ccpp_proj,
        ipress_proj,
        how="left",
        distance_col=f"dist_nearest_m_{label}",
    ).drop(columns="index_right", errors="ignore")

    # Drop duplicate index entries caused by equidistant ties (keep first)
    nearest = nearest[~nearest.index.duplicated(keep="first")]

    # Re-attach to original CCPP (keep EPSG:4326 geometry)
    result = ccpp.copy()
    result[f"dist_nearest_m_{label}"]   = nearest[f"dist_nearest_m_{label}"]
    result[f"nearest_ipress_{label}"]   = nearest["codigo_unico"]
    result[f"nearest_cat_{label}"]      = nearest["categoria"]

    valid = result[f"dist_nearest_m_{label}"].notna()
    print(f"  Nearest facility ({label}): "
          f"median {result.loc[valid, f'dist_nearest_m_{label}'].median()/1000:.1f} km | "
          f"max {result.loc[valid, f'dist_nearest_m_{label}'].max()/1000:.1f} km")
    return result


# ── District access metrics ───────────────────────────────────────────────────

def compute_district_access(
    ccpp_with_dist: gpd.GeoDataFrame,
    distritos: gpd.GeoDataFrame,
    label: str = "baseline",
) -> pd.DataFrame:
    """
    Aggregate CCPP nearest-facility distances to district level.

    Metrics per district:
    - n_ccpp                : number of populated centers in district
    - dist_mean_m           : mean distance to nearest facility (metres)
    - dist_median_m         : median distance
    - dist_max_m            : worst-case distance (most isolated centre)
    - pct_ccpp_gt5km        : % of centres > 5 km from any facility
    - pct_ccpp_gt20km       : % of centres > 20 km from any facility
    """
    dist_col = f"dist_nearest_m_{label}"

    ccpp_in_dist = assign_to_districts(
        ccpp_with_dist[ccpp_with_dist[dist_col].notna()].copy(),
        distritos,
        label=f"CCPP ({label})",
    )

    def pct_gt(series, threshold):
        return (series > threshold).mean() * 100

    access = (
        ccpp_in_dist.groupby("ubigeo_distrito")[dist_col]
        .agg(
            n_ccpp         = "count",
            dist_mean_m    = "mean",
            dist_median_m  = "median",
            dist_max_m     = "max",
            pct_ccpp_gt5km  = lambda s: pct_gt(s, 5_000),
            pct_ccpp_gt20km = lambda s: pct_gt(s, 20_000),
        )
        .reset_index()
        .rename(columns={"ubigeo_distrito": "ubigeo"})
    )

    # Round for readability
    for col in ["dist_mean_m", "dist_median_m", "dist_max_m"]:
        access[col] = access[col].round(0).astype("Int64")
    for col in ["pct_ccpp_gt5km", "pct_ccpp_gt20km"]:
        access[col] = access[col].round(2)

    access = access.add_suffix(f"_{label}").rename(
        columns={f"ubigeo_{label}": "ubigeo"}
    )

    print(f"  District access ({label}): {len(access):,} districts with CCPP data")
    return access


# ── District master GeoDataFrame ──────────────────────────────────────────────

def build_district_master(
    distritos: gpd.GeoDataFrame,
    supply: pd.DataFrame,
    access_baseline: pd.DataFrame,
    access_alternative: pd.DataFrame,
    district_susalud: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Join all district-level metrics into one GeoDataFrame.

    One row per district. Saved to data/processed/district_master.gpkg.

    Columns
    -------
    geometry                   : district polygon (EPSG:4326)
    ubigeo / iddpto / idprov   : identifiers
    Supply columns             : from compute_district_supply
    Access baseline columns    : from compute_district_access (baseline)
    Access alternative columns : from compute_district_access (alternative)
    SUSALUD columns            : latest-year emergency production per district
    """
    # Latest available year of SUSALUD data per district (include names for labelling)
    name_cols = [c for c in ["departamento", "provincia", "distrito"] if c in district_susalud.columns]
    latest = (
        district_susalud.sort_values("anho")
        .groupby("ubigeo")
        .last()
        .reset_index()
        [["ubigeo", "anho", "total_atenciones", "total_atendidos", "n_ipress_reportantes"] + name_cols]
        .rename(columns={
            "anho": "susalud_latest_year",
            "total_atenciones": "susalud_atenciones",
            "total_atendidos": "susalud_atendidos",
            "n_ipress_reportantes": "susalud_n_ipress",
        })
    )

    master = (
        distritos
        .merge(supply,             on="ubigeo", how="left")
        .merge(access_baseline,    on="ubigeo", how="left")
        .merge(access_alternative, on="ubigeo", how="left")
        .merge(latest,             on="ubigeo", how="left")
    )
    master["susalud_atenciones"]    = master["susalud_atenciones"].fillna(0).astype(int)
    master["susalud_atendidos"]     = master["susalud_atendidos"].fillna(0).astype(int)
    master["susalud_n_ipress"]      = master["susalud_n_ipress"].fillna(0).astype(int)

    print(f"  district_master: {len(master):,} districts × {len(master.columns)} columns")
    return master


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_geospatial_pipeline() -> dict:
    """
    Run the full Task 2 geospatial pipeline.
    Reads from data/processed/, writes district_master.gpkg and
    ccpp_with_distances.gpkg back to data/processed/.
    """
    print("\n=== Task 2 — Geospatial Integration ===\n")

    # ── Load cleaned datasets ────────────────────────────────────────────────
    print("── Loading processed datasets ──")
    ipress           = _gpkg("ipress_clean")
    ccpp             = _gpkg("ccpp_clean")
    distritos        = _gpkg("distritos_clean")
    facility_annual  = _parquet("susalud_facility_annual")
    district_susalud = _parquet("susalud_district_annual")
    print(f"  IPRESS: {len(ipress):,} | CCPP: {len(ccpp):,} | "
          f"Districts: {len(distritos):,}")

    # ── Emergency facility subsets ───────────────────────────────────────────
    print("\n── Defining emergency facility sets ──")
    ipress_baseline    = get_emergency_ipress_active(ipress, facility_annual)
    ipress_alternative = get_emergency_ipress_structural(ipress)

    # ── District supply ──────────────────────────────────────────────────────
    print("\n── Computing district supply metrics ──")
    supply = compute_district_supply(ipress, facility_annual, distritos)

    # ── CCPP nearest-facility distances ──────────────────────────────────────
    print("\n── Computing nearest-facility distances (baseline) ──")
    ccpp_dist = compute_nearest_facility(ccpp, ipress_baseline, label="baseline")

    print("\n── Computing nearest-facility distances (alternative) ──")
    ccpp_dist = compute_nearest_facility(ccpp_dist, ipress_alternative, label="alternative")

    # ── CCPP → district assignment + access aggregation ──────────────────────
    print("\n── Aggregating CCPP access metrics to districts ──")
    access_baseline    = compute_district_access(ccpp_dist, distritos, label="baseline")
    access_alternative = compute_district_access(ccpp_dist, distritos, label="alternative")

    # ── Build district master ────────────────────────────────────────────────
    print("\n── Building district master GeoDataFrame ──")
    master = build_district_master(
        distritos, supply,
        access_baseline, access_alternative,
        district_susalud,
    )

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\n── Saving outputs ──")
    master.to_file(PROCESSED_DIR / "district_master.gpkg", driver="GPKG")
    print(f"  Saved → district_master.gpkg")

    ccpp_save_cols = ["geometry", "longitud", "latitud",
                      "dist_nearest_m_baseline", "nearest_ipress_baseline",
                      "nearest_cat_baseline",
                      "dist_nearest_m_alternative", "nearest_ipress_alternative",
                      "nearest_cat_alternative"]
    ccpp_save = ccpp_dist[[c for c in ccpp_save_cols if c in ccpp_dist.columns]].copy()
    ccpp_save.to_file(PROCESSED_DIR / "ccpp_with_distances.gpkg", driver="GPKG")
    print(f"  Saved → ccpp_with_distances.gpkg")

    _print_summary(master)
    return {
        "master":   master,
        "ccpp_dist": ccpp_dist,
        "supply":   supply,
    }


def _print_summary(master: gpd.GeoDataFrame):
    print("\n=== Geospatial Summary ===")
    print(f"  Total districts                      : {len(master):,}")
    print(f"  Districts with ≥1 facility (baseline): "
          f"{(master['n_emergency_active'] > 0).sum():,}")
    print(f"  Districts with no facility (baseline): "
          f"{(master['n_emergency_active'] == 0).sum():,}")
    has_access = master["dist_median_m_baseline"].notna()
    print(f"  Districts with CCPP access data      : {has_access.sum():,}")
    if has_access.any():
        med = master.loc[has_access, "dist_median_m_baseline"].astype(float).median()
        print(f"  National median dist to facility     : {med/1000:.1f} km")
    print()


if __name__ == "__main__":
    run_geospatial_pipeline()
