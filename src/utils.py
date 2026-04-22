import chardet
import pandas as pd
from pathlib import Path


def detect_encoding(path: Path, sample_bytes: int = 50_000) -> str:
    """Detect the character encoding of a file using chardet."""
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    result = chardet.detect(raw)
    return result["encoding"] or "utf-8"


def pad_ubigeo(series: pd.Series) -> pd.Series:
    """Zero-pad UBIGEO codes to exactly 6 characters."""
    return series.astype(str).str.strip().str.zfill(6)
