"""
Cleaning and preprocessing — one function per dataset.
Each function returns a clean object ready for geospatial.py or metrics.py.
Cleaned outputs are saved to data/processed/.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
from pathlib import Path

from .utils import pad_ubigeo

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Codes SUSALUD uses when a facility did not report disaggregated data
_NE_CODES = {"NE_0001", "NE_0002"}

# Valid coordinate bounding box for Peru (WGS-84 decimal degrees)
_LON_MIN, _LON_MAX = -82.0, -68.0
_LAT_MIN, _LAT_MAX = -19.0,   0.0


# ── IPRESS ────────────────────────────────────────────────────────────────────

def clean_ipress(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Clean the IPRESS facility registry.

    Key decisions
    -------------
    - Keep only ACTIVADO facilities (removes ~700 deactivated records).
    - Coordinates: the raw columns 'NORTE' (renamed longitud) and 'ESTE'
      (renamed latitud) are mislabeled — they store longitude and latitude
      values respectively (verified against known district locations).
      Valid coordinate ranges for Peru: lon ∈ [-82, -68], lat ∈ [-19, 0].
    - Rows with coordinates outside those ranges get coords_valid=False;
      their geometry is set to None so spatial joins skip them safely.
    """
    df = df.copy()

    # Standardize free-text fields
    str_cols = ["departamento", "provincia", "distrito", "institucion",
                "clasificacion", "tipo", "estado", "situacion", "condicion",
                "categoria", "sector"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df["ubigeo"] = pad_ubigeo(df["ubigeo"])
    df["codigo_unico"] = df["codigo_unico"].astype(str).str.strip()

    # Filter: keep only active facilities
    n_raw = len(df)
    df = df[df["estado"] == "ACTIVADO"].copy()
    print(f"IPRESS: kept {len(df):,} / {n_raw:,} active facilities")

    # Cast numeric fields
    df["camas"] = pd.to_numeric(df["camas"], errors="coerce").fillna(0).astype(int)
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["altitud"] = pd.to_numeric(df["altitud"], errors="coerce")

    # Validate coordinates
    valid = (
        df["longitud"].between(_LON_MIN, _LON_MAX)
        & df["latitud"].between(_LAT_MIN, _LAT_MAX)
    )
    df["coords_valid"] = valid
    print(f"IPRESS: {valid.sum():,} facilities have valid coordinates "
          f"({valid.mean():.1%})")

    # Build geometry (None for facilities without valid coords)
    geom = gpd.points_from_xy(
        df["longitud"].where(valid),
        df["latitud"].where(valid),
    )
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    return gdf


# ── SUSALUD ───────────────────────────────────────────────────────────────────

def clean_susalud(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean SUSALUD emergency production data (2015–2026).

    Key decisions
    -------------
    - Rows where total_atenciones ∈ {NE_0001, NE_0002} are non-reporters
      (the IPRESS did not send disaggregated data for that period).
      These are dropped before aggregation — they would introduce zeros
      that undercount true activity.
    - The raw data is disaggregated by SEXO × EDAD × MES. We aggregate
      upward to two levels:
        1. facility_annual  (ubigeo + co_ipress + anho) — for supply analysis
        2. district_annual  (ubigeo + anho) — for district-level scoring

    Returns
    -------
    facility_annual : pd.DataFrame
    district_annual : pd.DataFrame
    """
    df = df.copy()

    # ── Filter non-reporters ──────────────────────────────────────────────────
    ne_mask = df["total_atenciones"].astype(str).isin(_NE_CODES)
    n_total = len(df)
    df = df[~ne_mask].copy()
    print(f"SUSALUD: kept {len(df):,} / {n_total:,} rows "
          f"({len(df)/n_total:.1%}) after dropping NE non-reporters")

    # ── Cast types ────────────────────────────────────────────────────────────
    df["total_atenciones"] = pd.to_numeric(df["total_atenciones"], errors="coerce").fillna(0).astype(int)
    df["total_atendidos"] = pd.to_numeric(df["total_atendidos"], errors="coerce").fillna(0).astype(int)
    df["anho"] = pd.to_numeric(df["anho"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")

    # ── Standardize text ──────────────────────────────────────────────────────
    for col in ["departamento", "provincia", "distrito", "sector", "categoria"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    df["ubigeo"] = pad_ubigeo(df["ubigeo"])
    df["co_ipress"] = df["co_ipress"].astype(str).str.strip()

    # ── Aggregate 1: facility × year ──────────────────────────────────────────
    facility_annual = (
        df.groupby(
            ["ubigeo", "anho", "co_ipress", "nombre",
             "sector", "categoria", "departamento", "provincia", "distrito"],
            as_index=False,
        )
        .agg(
            total_atenciones=("total_atenciones", "sum"),
            total_atendidos=("total_atendidos", "sum"),
            meses_activo=("mes", "nunique"),
        )
    )

    # ── Aggregate 2: district × year ─────────────────────────────────────────
    district_annual = (
        df.groupby(
            ["ubigeo", "anho", "departamento", "provincia", "distrito"],
            as_index=False,
        )
        .agg(
            total_atenciones=("total_atenciones", "sum"),
            total_atendidos=("total_atendidos", "sum"),
            n_ipress_reportantes=("co_ipress", "nunique"),
        )
    )

    print(f"SUSALUD: facility_annual → {len(facility_annual):,} rows | "
          f"district_annual → {len(district_annual):,} rows")
    return facility_annual, district_annual


# ── CCPP ──────────────────────────────────────────────────────────────────────

def clean_ccpp(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean the Centros Poblados (IGN 1:100 000) shapefile.

    Key decisions
    -------------
    - Normalize all column names to lowercase with underscores.
    - Reproject to WGS-84 (EPSG:4326) if the source CRS differs.
    - Drop rows with null or empty geometry.
    - Derive longitud / latitud from point geometry for later use in
      distance calculations.
    - Standardize the district UBIGEO field (6-digit zero-padded string).
    """
    gdf = gdf.copy()

    # Normalize column names
    gdf.columns = [c.strip().lower().replace(" ", "_") for c in gdf.columns]

    # Ensure WGS-84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Drop invalid geometry
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    print(f"CCPP: dropped {n_before - len(gdf):,} rows with invalid geometry; "
          f"{len(gdf):,} centers remain")

    # Extract coordinates from point geometry
    if gdf.geometry.geom_type.iloc[0] == "Point":
        gdf["longitud"] = gdf.geometry.x
        gdf["latitud"] = gdf.geometry.y

    # Standardize UBIGEO — look for any column whose name contains district code hints
    ubigeo_col = next(
        (c for c in gdf.columns if c in ("ubigeo", "iddist", "ubigdist", "cod_dist")),
        None,
    )
    if ubigeo_col and ubigeo_col != "ubigeo":
        gdf = gdf.rename(columns={ubigeo_col: "ubigeo"})
    if "ubigeo" in gdf.columns:
        gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])

    return gdf


# ── DISTRITOS ─────────────────────────────────────────────────────────────────

def clean_distritos(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean the district boundary shapefile.

    Key decisions
    -------------
    - Keep only IDDIST (renamed to ubigeo), IDDPTO, IDPROV, and geometry.
    - Reproject to WGS-84 for consistency with all other datasets.
    - UBIGEO zero-padded to 6 characters.
    """
    gdf = gdf.copy()
    gdf.columns = [c.strip().upper() for c in gdf.columns]

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.rename(columns={"IDDIST": "ubigeo", "IDDPTO": "iddpto", "IDPROV": "idprov"})
    gdf["ubigeo"] = pad_ubigeo(gdf["ubigeo"])

    keep = [c for c in ["ubigeo", "iddpto", "idprov", "geometry"] if c in gdf.columns]
    return gdf[keep]


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_processed(obj, filename: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """
    Save a cleaned DataFrame or GeoDataFrame to data/processed/.
    DataFrames → Parquet. GeoDataFrames → GeoPackage (.gpkg).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, gpd.GeoDataFrame):
        path = processed_dir / (Path(filename).stem + ".gpkg")
        obj.to_file(path, driver="GPKG")
    else:
        path = processed_dir / (Path(filename).stem + ".parquet")
        obj.to_parquet(path, index=False)

    print(f"Saved → {path}")
    return path


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_cleaning_pipeline(raw_dir: Path | None = None) -> dict:
    """
    Run the full Task 1 cleaning pipeline.
    Returns a dict with all cleaned objects for downstream use.
    """
    from .data_loader import (
        load_ipress, load_susalud, load_ccpp, load_distritos,
        RAW_DIR,
    )

    if raw_dir is None:
        raw_dir = RAW_DIR

    print("\n=== Task 1 — Data ingestion & cleaning ===\n")

    # Load
    print("── Loading raw data ──")
    ipress_raw = load_ipress(raw_dir)
    susalud_raw = load_susalud(raw_dir)
    ccpp_raw = load_ccpp(raw_dir)
    distritos_raw = load_distritos(raw_dir)

    # Clean
    print("\n── Cleaning ──")
    ipress = clean_ipress(ipress_raw)
    facility_annual, district_annual = clean_susalud(susalud_raw)
    ccpp = clean_ccpp(ccpp_raw)
    distritos = clean_distritos(distritos_raw)

    # Save
    print("\n── Saving to data/processed/ ──")
    save_processed(ipress, "ipress_clean")
    save_processed(facility_annual, "susalud_facility_annual")
    save_processed(district_annual, "susalud_district_annual")
    save_processed(ccpp, "ccpp_clean")
    save_processed(distritos, "distritos_clean")

    # Summary report
    _print_summary(ipress, facility_annual, district_annual, ccpp, distritos)

    return {
        "ipress": ipress,
        "facility_annual": facility_annual,
        "district_annual": district_annual,
        "ccpp": ccpp,
        "distritos": distritos,
    }


def _print_summary(ipress, facility_annual, district_annual, ccpp, distritos):
    print("\n=== Cleaning Summary ===")
    print(f"  IPRESS facilities (active)    : {len(ipress):>8,}")
    print(f"    · with valid coordinates    : {ipress['coords_valid'].sum():>8,}")
    print(f"    · without coordinates       : {(~ipress['coords_valid']).sum():>8,}")
    print(f"  SUSALUD facility-year rows    : {len(facility_annual):>8,}")
    print(f"  SUSALUD district-year rows    : {len(district_annual):>8,}")
    print(f"    · years covered             : {sorted(district_annual['anho'].dropna().unique().tolist())}")
    print(f"    · districts with activity   : {district_annual['ubigeo'].nunique():>8,}")
    print(f"  CCPP populated centers        : {len(ccpp):>8,}")
    print(f"  District polygons             : {len(distritos):>8,}")


if __name__ == "__main__":
    run_cleaning_pipeline()
