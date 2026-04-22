"""
Cleaning and preprocessing — one function per dataset.

Every filter, cast, or duplicate-drop is recorded in a CleaningLog so the
pipeline can write an auditable summary to output/tables/cleaning_report.md.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
from pathlib import Path

from .utils import pad_ubigeo

PROCESSED_DIR  = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_DIR     = Path(__file__).resolve().parents[1] / "output" / "tables"

# Codes SUSALUD uses when a facility did not report disaggregated data
_NE_CODES = {"NE_0001", "NE_0002"}

# Valid WGS-84 bounding box for Peru (decimal degrees)
_LON_MIN, _LON_MAX = -82.0, -68.0
_LAT_MIN, _LAT_MAX = -19.0,   0.0


# ── Cleaning log ──────────────────────────────────────────────────────────────

class CleaningLog:
    """Collects one entry per cleaning decision; writes a markdown report."""

    def __init__(self):
        self._entries: list[dict] = []

    def record(self, dataset: str, step: str, before: int, after: int, reason: str):
        self._entries.append({
            "Dataset":  dataset,
            "Step":     step,
            "Before":   before,
            "Dropped":  before - after,
            "After":    after,
            "Reason":   reason,
        })

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Cleaning & Filtering Decisions\n",
            "| Dataset | Step | Before | Dropped | After | Reason |",
            "|---------|------|-------:|--------:|------:|--------|",
        ]
        for e in self._entries:
            lines.append(
                f"| {e['Dataset']} | {e['Step']} "
                f"| {e['Before']:,} | {e['Dropped']:,} | {e['After']:,} "
                f"| {e['Reason']} |"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Cleaning report → {path}")


# ── IPRESS ────────────────────────────────────────────────────────────────────

def clean_ipress(df: pd.DataFrame, log: CleaningLog) -> gpd.GeoDataFrame:
    """
    Clean the IPRESS health facility registry.

    Decisions
    ---------
    1. Keep only ACTIVADO facilities.
    2. Drop duplicate codigo_unico — keep the row with the most information
       (non-null coordinates preferred).
    3. Remove rows with invalid coordinates but keep them in the GeoDataFrame
       with geometry=None so downstream joins can still use non-spatial fields.
    """
    df = df.copy()
    n0 = len(df)

    # ── Standardize text ──────────────────────────────────────────────────────
    str_cols = ["departamento", "provincia", "distrito", "institucion",
                "clasificacion", "tipo", "estado", "situacion",
                "condicion", "categoria"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df["ubigeo"]      = pad_ubigeo(df["ubigeo"])
    df["codigo_unico"] = df["codigo_unico"].astype(str).str.strip()

    # ── Step 1: Keep only active facilities ──────────────────────────────────
    df = df[df["estado"] == "ACTIVADO"].copy()
    log.record("IPRESS", "Keep ACTIVADO only", n0, len(df),
               "estado != 'ACTIVADO' — deactivated or closed facilities")

    # ── Cast numeric fields ───────────────────────────────────────────────────
    df["camas"]    = pd.to_numeric(df["camas"],    errors="coerce").fillna(0).astype(int)
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df["latitud"]  = pd.to_numeric(df["latitud"],  errors="coerce")
    df["altitud"]  = pd.to_numeric(df["altitud"],  errors="coerce")

    # ── Step 2: Remove duplicate codigo_unico ─────────────────────────────────
    # Sort so rows with valid coordinates come first, then keep first per key
    n_before = len(df)
    df["_has_coords"] = df["longitud"].notna() & df["latitud"].notna()
    df = df.sort_values("_has_coords", ascending=False)
    df = df.drop_duplicates(subset="codigo_unico", keep="first")
    df = df.drop(columns="_has_coords")
    log.record("IPRESS", "Drop duplicate codigo_unico", n_before, len(df),
               "Same facility registered more than once; kept row with coordinates")

    # ── Step 3: Validate coordinates ──────────────────────────────────────────
    # NOTE: raw columns 'NORTE'/'ESTE' are mislabeled — 'NORTE' holds longitude
    # values and 'ESTE' holds latitude values (confirmed against known locations).
    valid = (
        df["longitud"].between(_LON_MIN, _LON_MAX)
        & df["latitud"].between(_LAT_MIN, _LAT_MAX)
    )
    df["coords_valid"] = valid
    log.record("IPRESS", "Flag invalid coordinates", len(df),
               len(df),   # rows not dropped — flagged only
               f"Coords outside Peru bbox [lon {_LON_MIN},{_LON_MAX}] "
               f"[lat {_LAT_MIN},{_LAT_MAX}]; kept with geometry=None")

    # ── Build GeoDataFrame ────────────────────────────────────────────────────
    geom = gpd.points_from_xy(
        df["longitud"].where(valid),
        df["latitud"].where(valid),
    )
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    print(f"  IPRESS → {len(gdf):,} facilities "
          f"({valid.sum():,} with valid coords, "
          f"{(~valid).sum():,} without)")
    return gdf


# ── SUSALUD ───────────────────────────────────────────────────────────────────

def clean_susalud(
    df: pd.DataFrame, log: CleaningLog
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean SUSALUD emergency production data (2015–2026).

    Decisions
    ---------
    1. Drop rows where total_atenciones ∈ {NE_0001, NE_0002}: these are
       non-reporters that did not submit disaggregated data. Treating them
       as zeros would undercount true activity.
    2. Drop exact duplicates on the grain (ubigeo, anho, mes, co_ipress,
       sexo, edad) — can occur when a file was uploaded more than once.
    3. Aggregate raw grain (sexo × edad × mes) up to two levels:
       - facility_annual  → for supply-side analysis
       - district_annual  → for district scoring
    """
    df = df.copy()
    n0 = len(df)

    # ── Step 1: Drop NE non-reporters ─────────────────────────────────────────
    ne_mask = df["total_atenciones"].astype(str).isin(_NE_CODES)
    df = df[~ne_mask].copy()
    log.record("SUSALUD", "Drop NE non-reporters", n0, len(df),
               "total_atenciones ∈ {NE_0001, NE_0002}: facility did not report")

    # ── Cast types ────────────────────────────────────────────────────────────
    df["total_atenciones"] = pd.to_numeric(df["total_atenciones"], errors="coerce").fillna(0).astype(int)
    df["total_atendidos"]  = pd.to_numeric(df["total_atendidos"],  errors="coerce").fillna(0).astype(int)
    df["anho"] = pd.to_numeric(df["anho"], errors="coerce").astype("Int64")
    df["mes"]  = pd.to_numeric(df["mes"],  errors="coerce").astype("Int64")

    # ── Standardize text ──────────────────────────────────────────────────────
    for col in ["departamento", "provincia", "distrito", "sector", "categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    df["ubigeo"]    = pad_ubigeo(df["ubigeo"])
    df["co_ipress"] = df["co_ipress"].astype(str).str.strip()

    # ── Step 2: Drop exact duplicates on data grain ───────────────────────────
    grain = ["ubigeo", "anho", "mes", "co_ipress", "sexo", "edad"]
    n_before = len(df)
    df = df.drop_duplicates(subset=grain, keep="first")
    log.record("SUSALUD", "Drop duplicate grain rows", n_before, len(df),
               f"Exact duplicate on ({', '.join(grain)}): likely double-upload")

    # ── Aggregate 1: facility × year ──────────────────────────────────────────
    facility_annual = (
        df.groupby(
            ["ubigeo", "anho", "co_ipress", "nombre",
             "sector", "categoria", "departamento", "provincia", "distrito"],
            as_index=False,
        )
        .agg(
            total_atenciones  = ("total_atenciones", "sum"),
            total_atendidos   = ("total_atendidos",  "sum"),
            meses_activo      = ("mes", "nunique"),
        )
    )

    # ── Aggregate 2: district × year ──────────────────────────────────────────
    district_annual = (
        df.groupby(
            ["ubigeo", "anho", "departamento", "provincia", "distrito"],
            as_index=False,
        )
        .agg(
            total_atenciones    = ("total_atenciones", "sum"),
            total_atendidos     = ("total_atendidos",  "sum"),
            n_ipress_reportantes = ("co_ipress", "nunique"),
        )
    )

    log.record("SUSALUD", "Aggregate to facility×year", len(df),
               len(facility_annual),
               "Sum total_atenciones/atendidos across sexo × edad × mes per facility per year")
    log.record("SUSALUD", "Aggregate to district×year", len(df),
               len(district_annual),
               "Sum total_atenciones/atendidos across all facilities per district per year")

    print(f"  SUSALUD → facility_annual: {len(facility_annual):,} rows | "
          f"district_annual: {len(district_annual):,} rows")
    return facility_annual, district_annual


# ── CCPP ──────────────────────────────────────────────────────────────────────

def clean_ccpp(gdf: gpd.GeoDataFrame, log: CleaningLog) -> gpd.GeoDataFrame:
    """
    Clean the Centros Poblados shapefile (IGN 1:100 000).

    Decisions
    ---------
    1. Reproject to WGS-84 (EPSG:4326).
    2. Drop rows with null or empty geometry.
    3. Drop duplicate populated centers on exact (longitud, latitud) pair.
    4. Standardize UBIGEO to 6-digit string.
    """
    gdf = gdf.copy()
    n0 = len(gdf)

    # Normalize column names
    gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]

    # ── Reproject to WGS-84 ───────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # ── Step 1: Drop null/empty geometry ─────────────────────────────────────
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    log.record("CCPP", "Drop null/empty geometry", n_before, len(gdf),
               "Rows where geometry is None or empty — cannot be placed on map")

    # Extract lon/lat from point geometry
    if gdf.geometry.geom_type.iloc[0] == "Point":
        gdf["longitud"] = gdf.geometry.x
        gdf["latitud"]  = gdf.geometry.y

    # ── Step 2: Drop duplicate coordinates ───────────────────────────────────
    n_before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["longitud", "latitud"], keep="first")
    log.record("CCPP", "Drop duplicate coordinates", n_before, len(gdf),
               "Two or more populated centers at identical (longitud, latitud)")

    # ── Standardize UBIGEO ────────────────────────────────────────────────────
    ubigeo_col = next(
        (c for c in gdf.columns if c in ("ubigeo", "iddist", "ubigdist", "cod_dist")),
        None,
    )
    if ubigeo_col and ubigeo_col != "ubigeo":
        gdf = gdf.rename(columns={ubigeo_col: "ubigeo"})
    if "ubigeo" in gdf.columns:
        gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])

    print(f"  CCPP → {len(gdf):,} populated centers (dropped {n0 - len(gdf):,})")
    return gdf


# ── DISTRITOS ─────────────────────────────────────────────────────────────────

def clean_distritos(gdf: gpd.GeoDataFrame, log: CleaningLog) -> gpd.GeoDataFrame:
    """
    Clean the district boundary shapefile.

    Decisions
    ---------
    1. Reproject to WGS-84 (EPSG:4326).
    2. Drop duplicate UBIGEO polygons (keep largest area, most representative).
    3. Keep only ubigeo + iddpto + idprov + geometry.
    """
    gdf = gdf.copy()

    # Detect geometry column before uppercasing (pyogrio loads it as 'GEOMETRY')
    geom_col = gdf.geometry.name
    gdf.columns = [c.strip().upper() for c in gdf.columns]
    geom_col_upper = geom_col.upper()
    gdf = gdf.set_geometry(geom_col_upper)

    # ── Reproject ─────────────────────────────────────────────────────────────
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.rename(columns={
        "IDDIST": "ubigeo", "IDDPTO": "iddpto", "IDPROV": "idprov",
        geom_col_upper: "geometry",
    })
    gdf = gdf.set_geometry("geometry")
    gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])

    # ── Step 1: Drop duplicate UBIGEO ────────────────────────────────────────
    n_before = len(gdf)
    # Compute area in meters (EPSG:24891 = Peru UTM) to pick the largest polygon
    gdf["_area"] = gdf.geometry.to_crs("EPSG:24891").area
    gdf = gdf.sort_values("_area", ascending=False).drop_duplicates(
        subset="ubigeo", keep="first"
    )
    gdf = gdf.drop(columns="_area")
    log.record("DISTRITOS", "Drop duplicate UBIGEO", n_before, len(gdf),
               "Same UBIGEO with multiple polygons; kept largest-area polygon")

    keep = [c for c in ["ubigeo", "iddpto", "idprov", "geometry"] if c in gdf.columns]
    gdf = gdf[keep]

    print(f"  DISTRITOS → {len(gdf):,} districts")
    return gdf


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_processed(obj, filename: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Save DataFrame → Parquet, GeoDataFrame → GeoPackage."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, gpd.GeoDataFrame):
        path = processed_dir / (Path(filename).stem + ".gpkg")
        obj.to_file(path, driver="GPKG")
    else:
        path = processed_dir / (Path(filename).stem + ".parquet")
        obj.to_parquet(path, index=False)

    print(f"  Saved → {path.name}")
    return path


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_cleaning_pipeline(raw_dir: Path | None = None) -> dict:
    """
    Run the full Task 1 cleaning pipeline and persist all outputs.
    Returns a dict of cleaned objects for downstream use.
    """
    from .data_loader import (
        load_ipress, load_susalud, load_ccpp, load_distritos, RAW_DIR,
    )

    if raw_dir is None:
        raw_dir = RAW_DIR

    log = CleaningLog()

    print("\n=== Task 1 — Data Ingestion & Cleaning ===\n")

    # ── Load raw ──────────────────────────────────────────────────────────────
    print("── Loading raw datasets ──")
    ipress_raw    = load_ipress(raw_dir)
    susalud_raw   = load_susalud(raw_dir)
    ccpp_raw      = load_ccpp(raw_dir)
    distritos_raw = load_distritos(raw_dir)

    # ── Clean ─────────────────────────────────────────────────────────────────
    print("\n── Cleaning ──")
    ipress                          = clean_ipress(ipress_raw, log)
    facility_annual, district_annual = clean_susalud(susalud_raw, log)
    ccpp                            = clean_ccpp(ccpp_raw, log)
    distritos                       = clean_distritos(distritos_raw, log)

    # ── Save cleaned datasets ─────────────────────────────────────────────────
    print("\n── Saving cleaned datasets ──")
    save_processed(ipress,           "ipress_clean")
    save_processed(facility_annual,  "susalud_facility_annual")
    save_processed(district_annual,  "susalud_district_annual")
    save_processed(ccpp,             "ccpp_clean")
    save_processed(distritos,        "distritos_clean")

    # ── Save cleaning report ──────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "cleaning_report.md"
    log.save(report_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(ipress, facility_annual, district_annual, ccpp, distritos)

    return {
        "ipress":           ipress,
        "facility_annual":  facility_annual,
        "district_annual":  district_annual,
        "ccpp":             ccpp,
        "distritos":        distritos,
    }


def _print_summary(ipress, facility_annual, district_annual, ccpp, distritos):
    years = sorted(district_annual["anho"].dropna().unique().tolist())
    print("\n=== Final Summary ===")
    print(f"  IPRESS facilities             : {len(ipress):>8,}")
    print(f"    · with valid coordinates    : {ipress['coords_valid'].sum():>8,}  ({ipress['coords_valid'].mean():.1%})")
    print(f"    · without valid coordinates : {(~ipress['coords_valid']).sum():>8,}")
    print(f"  SUSALUD facility-year records : {len(facility_annual):>8,}")
    print(f"  SUSALUD district-year records : {len(district_annual):>8,}")
    print(f"    · years covered             : {years}")
    print(f"    · districts with activity   : {district_annual['ubigeo'].nunique():>8,}")
    print(f"  CCPP populated centers        : {len(ccpp):>8,}")
    print(f"  District polygons             : {len(distritos):>8,}")
    print()


if __name__ == "__main__":
    run_cleaning_pipeline()
