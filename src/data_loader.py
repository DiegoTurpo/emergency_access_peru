"""
Raw data loaders — one function per source dataset.
All loaders return data in its original form with minimal transformation.
Column renaming and cleaning is handled in cleaning.py.
"""

import zipfile
import pandas as pd
import geopandas as gpd
from pathlib import Path

from .utils import detect_encoding

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# ── Column name maps ──────────────────────────────────────────────────────────

# IPRESS has accented Spanish headers; rename by position to avoid encoding issues
IPRESS_COLS = [
    "institucion", "codigo_unico", "nombre", "clasificacion", "tipo",
    "departamento", "provincia", "distrito", "ubigeo", "direccion",
    "codigo_disa", "codigo_red", "codigo_microrred", "disa", "red", "microrred",
    "codigo_ue", "unidad_ejecutora", "categoria", "telefono",
    "tipo_doc_categorizacion", "nro_doc_categorizacion", "horario",
    "inicio_actividad", "director", "estado", "situacion", "condicion",
    "inspeccion",
    "longitud",   # raw column name "NORTE" — stores longitude values for Peru
    "latitud",    # raw column name "ESTE"  — stores latitude values for Peru
    "altitud",    # COTA (meters above sea level)
    "camas",
]

SUSALUD_COL_MAP = {
    "ANHO": "anho",
    "MES": "mes",
    "UBIGEO": "ubigeo",
    "DEPARTAMENTO": "departamento",
    "PROVINCIA": "provincia",
    "DISTRITO": "distrito",
    "SECTOR": "sector",
    "CATEGORIA": "categoria",
    "CO_IPRESS": "co_ipress",
    "RAZON_SOC": "nombre",
    "SEXO": "sexo",
    "EDAD": "edad",
    "NRO_TOTAL_ATENCIONES": "total_atenciones",
    "NRO_TOTAL_ATENDIDOS": "total_atendidos",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ipress(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load IPRESS health facility registry (MINSA open data)."""
    path = raw_dir / "IPRESS.csv"
    enc = detect_encoding(path)
    df = pd.read_csv(path, encoding=enc, dtype={"UBIGEO": str})
    df.columns = IPRESS_COLS
    return df


def load_susalud(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load and concatenate all annual SUSALUD emergency production files.
    Files cover 2015–2026 (partial). Semicolon-delimited.
    Rows with NE_0001 / NE_0002 codes are non-reporters — kept here,
    filtered in cleaning.py.
    """
    susalud_dir = raw_dir / "SUSALUD"
    files = sorted(susalud_dir.glob("ConsultaC1_*.csv"))
    if not files:
        raise FileNotFoundError(f"No SUSALUD files found in {susalud_dir}")

    dfs = []
    for f in files:
        enc = detect_encoding(f)
        df = pd.read_csv(f, encoding=enc, sep=";", dtype={"UBIGEO": str})
        df = df.rename(columns=SUSALUD_COL_MAP)
        # Tag source year from filename so it survives concat
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"SUSALUD loaded: {len(combined):,} rows from {len(files)} files")
    return combined


def load_ccpp(raw_dir: Path = RAW_DIR) -> gpd.GeoDataFrame:
    """
    Load Centros Poblados shapefile (IGN 1:100 000).
    Extracts CCPP_0.zip on first call; subsequent calls reuse extracted files.
    """
    zip_path = raw_dir / "CCPP_0.zip"
    extract_dir = raw_dir / "CCPP_extracted"

    if not extract_dir.exists():
        print("Extracting CCPP_0.zip …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found after extracting {zip_path}")

    gdf = gpd.read_file(shp_files[0])
    print(f"CCPP loaded: {len(gdf):,} populated centers | CRS: {gdf.crs}")
    return gdf


def load_distritos(raw_dir: Path = RAW_DIR) -> gpd.GeoDataFrame:
    """Load Peru district-level boundary shapefile."""
    path = raw_dir / "DISTRITOS.shp"
    gdf = gpd.read_file(path)
    print(f"DISTRITOS loaded: {len(gdf):,} districts | CRS: {gdf.crs}")
    return gdf
